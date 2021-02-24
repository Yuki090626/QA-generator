from django.contrib import admin
from generator.models import QAC, Document, Context

# Register your models here.
admin.site.register(QAC)
admin.site.register(Context)
admin.site.register(Document)