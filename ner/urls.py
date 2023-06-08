from django.urls import path, include, re_path
# from django.conf.urls import url

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    re_path(r'^get_named_entities/$', views.get_named_entities, name='get_named_entities'),
]