from django.urls import path

from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.index, name="index"),
    path('index.html', views.index, name="index"),
    
    #path('stock', views.render_plot_page),


]