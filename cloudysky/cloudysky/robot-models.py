from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import FileExtensionValidator
from django.utils import timezone
from datetime import timedelta

class User(AbstractUser):
    class Role(models.TextChoices):
        SERF  = 'serf',  'Serf'
        ADMIN = 'admin', 'Administrator'
    
    role        = models.CharField(max_length=10, choices=Role.choices, default=Role.SERF)
    bio         = models.TextField(blank=True)
    last_active = models.DateTimeField(auto_now=True)
    
    def is_admin(self):
        return self.role == self.Role.ADMIN


class Avatar(models.Model):
    avatar_img = models.ImageField(
        upload_to='avatars/',
        default='avatars/default.png',
        validators=[FileExtensionValidator(allowed_extensions=['png','jpg','jpeg','gif'])]
    )
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='avatar'
    )
    
    def __str__(self):
        return f"Avatar for {self.user.username}"


class SuppressionReason(models.Model):
    reason      = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.reason


class Media(models.Model):
    class MediaType(models.TextChoices):
        IMAGE = 'image', 'Image'
        VIDEO = 'video', 'Video'
        OTHER = 'other', 'Other'
    
    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name='media')
    file        = models.FileField(
        upload_to='media/',
        validators=[FileExtensionValidator(
            allowed_extensions=['jpg','jpeg','png','gif','mp4','pdf']
        )]
    )
    media_type  = models.CharField(max_length=10, choices=MediaType.choices)
    size        = models.PositiveIntegerField()  # in bytes
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Attach each upload to *either* a post or a comment
    post    = models.ForeignKey(
        'Post',
        null=True, blank=True,
        on_delete=models.CASCADE,
        related_name='media'
    )
    comment = models.ForeignKey(
        'Comment',
        null=True, blank=True,
        on_delete=models.CASCADE,
        related_name='media'
    )
    
    def save(self, *args, **kwargs):
        if not self.pk:
            self.size = self.file.size
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Media {self.id} by {self.user.username}"


class Post(models.Model):
    user          = models.ForeignKey(User,   on_delete=models.CASCADE, related_name='posts')
    content       = models.TextField()
    created_at    = models.DateTimeField(auto_now_add=True)
    is_hidden     = models.BooleanField(default=False)
    hidden_reason = models.ForeignKey(
        SuppressionReason,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='posts'
    )
    hidden_at     = models.DateTimeField(null=True, blank=True)
    
    def excerpt(self, length=100):
        return self.content[:length] + ('...' if len(self.content) > length else '')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Post {self.id} by {self.user.username}"


class Comment(models.Model):
    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comments')
    post          = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    content       = models.TextField(max_length=500)
    created_at    = models.DateTimeField(auto_now_add=True)
    is_hidden     = models.BooleanField(default=False)
    hidden_reason = models.ForeignKey(
        SuppressionReason,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='comments'
    )
    hidden_at     = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"Comment {self.id} by {self.user.username}"


class UserActivity(models.Model):
    user             = models.OneToOneField(User, on_delete=models.CASCADE, related_name='activity_stats')
    last_updated     = models.DateTimeField(auto_now=True)
    
    posts_count_1d   = models.PositiveIntegerField(default=0)
    posts_count_7d   = models.PositiveIntegerField(default=0)
    posts_count_30d  = models.PositiveIntegerField(default=0)
    
    comments_count_1d  = models.PositiveIntegerField(default=0)
    comments_count_7d  = models.PositiveIntegerField(default=0)
    comments_count_30d = models.PositiveIntegerField(default=0)
    
    suppressed_count_1d   = models.PositiveIntegerField(default=0)
    suppressed_count_7d   = models.PositiveIntegerField(default=0)
    suppressed_count_30d  = models.PositiveIntegerField(default=0)
    
    bytes_used_1d   = models.PositiveBigIntegerField(default=0)
    bytes_used_7d   = models.PositiveBigIntegerField(default=0)
    bytes_used_30d  = models.PositiveBigIntegerField(default=0)
    
    def update_stats(self):
        now = timezone.now()
        
        self.posts_count_1d  = self.user.posts.filter(created_at__gte=now - timedelta(days=1)).count()
        self.posts_count_7d  = self.user.posts.filter(created_at__gte=now - timedelta(days=7)).count()
        self.posts_count_30d = self.user.posts.filter(created_at__gte=now - timedelta(days=30)).count()
        
        self.comments_count_1d  = self.user.comments.filter(created_at__gte=now - timedelta(days=1)).count()
        self.comments_count_7d  = self.user.comments.filter(created_at__gte=now - timedelta(days=7)).count()
        self.comments_count_30d = self.user.comments.filter(created_at__gte=now - timedelta(days=30)).count()
        
        self.suppressed_count_1d  = (
             self.user.posts.filter(is_hidden=True, hidden_at__gte=now - timedelta(days=1)).count() +
             self.user.comments.filter(is_hidden=True, hidden_at__gte=now - timedelta(days=1)).count()
        )
        self.suppressed_count_7d  = (
             self.user.posts.filter(is_hidden=True, hidden_at__gte=now - timedelta(days=7)).count() +
             self.user.comments.filter(is_hidden=True, hidden_at__gte=now - timedelta(days=7)).count()
        )
        self.suppressed_count_30d = (
             self.user.posts.filter(is_hidden=True, hidden_at__gte=now - timedelta(days=30)).count() +
             self.user.comments.filter(is_hidden=True, hidden_at__gte=now - timedelta(days=30)).count()
        )
        
        self.bytes_used_1d   = sum(m.size for m in self.user.media.filter(uploaded_at__gte=now - timedelta(days=1)))
        self.bytes_used_7d   = sum(m.size for m in self.user.media.filter(uploaded_at__gte=now - timedelta(days=7)))
        self.bytes_used_30d  = sum(m.size for m in self.user.media.filter(uploaded_at__gte=now - timedelta(days=30)))
        
        self.last_updated = now
        self.save()
