from django.contrib import admin
from django.urls import path
from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name="index"),
    path('app/time', views.current_time, name='current_time'),
    path('app/sum', views.sum_numbers, name='sum_numbers'),
    path('index.html', views.index, name="index_html"),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='registration/login.html')),
    path('logout/', LogoutView.as_view(next_page='index'), name='logout'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/profile/', views.index, name='profile'),
    path('app/new', views.new_user_form, name='new_user_form'),
    path('app/createUser', views.create_user, name='create_user'),
    path("app/createPost", views.create_post, name="create_post"),
    path("app/createComment", views.create_comment, name="create_comment"),
    path("app/hidePost", views.hide_post, name="hide_post"),
    path("app/hideComment", views.hide_comment, name="hide_comment"),
    path('app/get_suppression_reasons', views.get_suppression_reasons, name='get_suppression_reasions'),
    path("app/new_post", views.new_post, name="new_post"),
    path("app/new_comment", views.new_comment, name="new_comment"),
    path('app/dumpFeed', views.dumpFeed, name='dumpFeed'),
    path('app/feed', views.feed, name='feed'),
    path('app/post/<int:post_id>', views.post_detail, name='post_detail'),
    path('app/feed_page', views.feed_page, name='feed_page'),
    path('app/post_page/<int:post_id>', views.post_page, name='post_page'),
]
