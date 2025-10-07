from django.http import (
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotAllowed,
    JsonResponse
)
import json
from django.shortcuts import render, redirect 
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from .models import Post, Comment, SuppressionReason
from django.db.models import Q

User = get_user_model()

def dummypage(request):
    if request.method == "GET":
        return HttpResponse("No content here, sorry!")
    return HttpResponseNotAllowed(["GET"])

def current_time(request):
    # return server time in CDT
    now_utc = timezone.now()
    cdt = now_utc - timedelta(hours=5)
    return HttpResponse(cdt.strftime("%H:%M"))


def sum_numbers(request):
    n1 = request.GET.get("n1", "0")
    n2 = request.GET.get("n2", "0")
    try:
        total = float(n1) + float(n2)
        if total.is_integer():
            return HttpResponse(str(int(total)))
        else:
            # trim trailing zeros
            s = f"{total:.10f}".rstrip("0").rstrip(".")
            return HttpResponse(s)
    except (ValueError, TypeError):
        return HttpResponse("Invalid numbers", status=400)


def index(request):
    now = timezone.now().strftime("%Y-%m-%d %H:%M:%S")
    bios = [
        "Ethan: I love sports and my friends.",
        "Jared: I love traveling, hiking, and learning languages.",
    ]
    return render(request, "app/index.html", {
        "bios": bios,
        "current_time": now,
    })


def new_user_form(request):
    if request.method == "POST":
        return create_user(request)
    return render(request, 'app/new_user_form.html', {'user': None, 'error': None})


@csrf_exempt
def create_user(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")
    username = request.POST.get("user_name")
    email = request.POST.get("email")
    password = request.POST.get("password")
    is_admin = request.POST.get("is_admin") == "true"

    if not (username and email and password):
        return render(request, 'app/new_user_form.html', {
            'error': 'Missing required fields',
            'user': None
        })

    if User.objects.filter(username=username).exists():
        return render(request, 'app/new_user_form.html', {
            'error': 'Username already exists',
            'user': None
        })
    if User.objects.filter(email=email).exists():
        return render(request, 'app/new_user_form.html', {
            'error': 'Email already in use',
            'user': None
        })

    try:
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            is_staff=is_admin,
            is_superuser=is_admin,
            role=(User.Role.ADMIN if is_admin else User.Role.SERF)
        )
        return render(request, 'app/new_user_form.html', {
            'user': user,
            'error': None
        })
    except Exception as e:
        return render(request, 'app/new_user_form.html', {
            'error': f"Error creating user: {e}",
            'user': None
        })


@csrf_exempt
def create_post(request):
    if not request.user.is_authenticated:
        return HttpResponse(status=401)
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    title   = request.POST.get("title")
    content = request.POST.get("content")
    if not (title and content):
        return HttpResponseBadRequest("Missing title or content")

    post = Post.objects.create(
        author=request.user,
        title=title,
        content=content,
    )

    return JsonResponse({
        "message":  "Successfully created post",
        "post_id":  post.id,
        "title":    post.title,
        "content":  post.content,
    })


@csrf_exempt
def create_comment(request):
    if not request.user.is_authenticated:
        return HttpResponse(status=401)
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    pid     = request.POST.get("post_id")
    content = request.POST.get("content")
    if not (pid and content):
        return HttpResponseBadRequest("Missing post_id or content")

    post = get_object_or_404(Post, id=pid)
    comment = Comment.objects.create(
        author=request.user,
        post=post,
        content=content,
    )

    return JsonResponse({
        "message":    "Successfully created comment",
        "comment_id": comment.id,
        "post_id":    post.id,
        "content":    comment.content,
    })

ALLOWED_SUPPRESSION_REASONS = ["spam", "abuse", "off-topic", "inappropriate", "other"]

@csrf_exempt
def get_suppression_reasons(request):
    """Return list of allowed suppression reasons"""
    return JsonResponse({
        "status": "ok",
        "reasons": ALLOWED_SUPPRESSION_REASONS
    })

@csrf_exempt
def hide_post(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body) if request.content_type == 'application/json' else request.POST
        pid = data.get("post_id")
        reason_str = data.get("reason")
        
        if not pid or not reason_str:
            return JsonResponse({"error": "Missing post_id or reason"}, status=400)
        
        reason_str = reason_str.strip().lower()
        
        if reason_str not in ALLOWED_SUPPRESSION_REASONS:
            return JsonResponse({
                "error": f"Invalid suppression reason '{reason_str}'",
                "available_reasons": ALLOWED_SUPPRESSION_REASONS
            }, status=400)

        reason_obj, created= SuppressionReason.objects.get_or_create(reason=reason_str)

        post = get_object_or_404(Post, id=int(pid))
        post.is_hidden = not post.is_hidden
        post.hidden_reason = reason_obj if post.is_hidden else None
        post.hidden_at = timezone.now() if post.is_hidden else None
        post.save()

        return JsonResponse({"status": "ok"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def hide_comment(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body) if request.content_type == 'application/json' else request.POST
        cid = data.get("comment_id")
        reason_str = data.get("reason")
        
        if not cid or not reason_str:
            return JsonResponse({"error": "Missing comment_id or reason"}, status=400)

        reason_str = reason_str.strip().lower()

        if reason_str not in ALLOWED_SUPPRESSION_REASONS:
            return JsonResponse({
                "error": f"Invalid suppression reason '{reason_str}'",
                "available_reasons": ALLOWED_SUPPRESSION_REASONS
            }, status=400)

        reason_obj, created = SuppressionReason.objects.get_or_create(reason=reason_str)

        comment = get_object_or_404(Comment, id=int(cid))
        comment.is_hidden = not comment.is_hidden
        comment.hidden_reason = reason_obj if comment.is_hidden else None
        comment.hidden_at = timezone.now() if comment.is_hidden else None
        comment.save()

        return JsonResponse({"status": "ok"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def new_post(request):
    if not request.user.is_authenticated:
        return HttpResponse(status=401)
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    return render(request, "app/new_post.html")


def new_comment(request):
    if not request.user.is_authenticated:
        return HttpResponse(status=401)
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    return render(request, "app/new_comment.html")


def public_feed(request):
    posts = Post.objects.filter(is_hidden=False).order_by('-created_at')

    if request.headers.get('Accept') == 'application/json':
        feed = []
        for post in posts:
            feed.append({
                "id": post.id,
                "title": post.title,
                "content": post.content,
                "username": post.author.username,
                "date": post.created_at.strftime("%Y-%m-%d %H:%M"),
                "is_hidden": post.is_hidden,
                "comments": []
            })
        return JsonResponse(feed, safe=False)
    
    return render(request, 'app/feed.html', {'is_public': True})


def dumpFeed(request):
    feed = []
    user = request.user
    is_admin = user.is_authenticated and user.is_staff

    for p in Post.objects.all().order_by("-created_at"):
        is_author = user.is_authenticated and p.author == user
        is_suppressed = p.is_hidden

        if not is_suppressed or is_admin or is_author:
            comments = []
            for c in p.comments.all().order_by("created_at"):
                comment_author    = user.is_authenticated and c.author == user
                comment_suppressed = c.is_hidden

                if not comment_suppressed or is_admin or comment_author:
                    comments.append({
                        "id":          c.id,
                        "username":    c.author.username,
                        "date":        c.created_at.strftime("%Y-%m-%d %H:%M"),
                        "content":     c.content,
                        "is_suppressed": comment_suppressed,
                    })

            feed.append({
                "id":           p.id,
                "username":     p.author.username,
                "date":         p.created_at.strftime("%Y-%m-%d %H:%M"),
                "title":        p.title,
                "content":      p.content,
                "is_suppressed": is_suppressed,
                "comments":     comments,
            })

    return JsonResponse(feed, safe=False)

def feed(request):
    if not request.user.is_authenticated:
        return HttpResponse(status=401)

    user = request.user
    qs   = Post.objects.all().order_by('-created_at')
    out  = []
    for p in qs:
        # skip suppressed posts unless you’re the author or an admin
        if p.is_hidden and not (user.is_admin() or p.author == user):
            continue

        out.append({
            "id":       p.id,
            "title":    p.title,
            "date":     p.created_at.strftime("%Y-%m-%d %H:%M"),
            "username": p.author.username,
            "content":  p.excerpt(100),  # first 100 chars
        })
    return JsonResponse(out, safe=False)


def post_detail(request, post_id):
    user = request.user

    try:
        post = Post.objects.get(id=post_id)
    except Post.DoesNotExist:
        return JsonResponse({"error": "Post not found"}, status=404)

    if post.is_hidden and not (user.is_authenticated and (user.is_admin() or post.author == user)):
        return JsonResponse({"error": "You do not have permission to view this post"}, status=403)

    data = {
        "id": post.id,
        "title": post.title,
        "date": post.created_at.strftime("%Y-%m-%d %H:%M"),
        "username": post.author.username,
        "content": post.content,
        "is_hidden": post.is_hidden,
        "comments": []
    }

    for comment in post.comments.all().order_by('created_at'):
        can_view_comment = not comment.is_hidden or (
            user.is_authenticated and (user.is_admin() or comment.author == user)
        )
        data["comments"].append({
            "id": comment.id,
            "username": comment.author.username,
            "date": comment.created_at.strftime("%Y-%m-%d %H:%M"),
            "content": "This comment has been removed" if not can_view_comment else comment.content,
            "is_hidden": comment.is_hidden
        })

    return JsonResponse(data)


def feed_page(request):
    if request.user.is_authenticated:
        return render(request, "app/feed.html", {'is_public': False})
    return public_feed(request)


def post_page(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    user = request.user
    
    if post.is_hidden and not (user.is_staff or post.author == user):
        return HttpResponse(status=404)
    
    context = {
        'post_id': post_id,
        'is_admin': user.is_staff,
        'username': user.username,
    }
    return render(request, "app/post_detail.html", context)