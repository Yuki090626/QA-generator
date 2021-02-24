from django.db import models
from django.conf import settings
import os


def delete_previous_file(function):
    def wrapper(*args, **kwargs):
        self = args[0]

        result = Document.objects.filter(pk=self.pk)
        previous = result[0] if len(result) else None
        super(Document, self).save()

        result = function(*args, **kwargs)

        if previous:
            os.remove(settings.MEDIA_ROOT + '/' + previous.document.name)
        return result
    return wrapper


class Document(models.Model):
    @delete_previous_file
    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        super(Document, self).save()

    @delete_previous_file
    def delete(self, using=None, keep_parents=False):
        super(Document, self).delete()

    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Context(models.Model):
    context = models.CharField('ドキュメント', max_length=1024)

    def __str__(self):
        return self.context


class QAC(models.Model):
    idx = models.IntegerField('ID', blank=True, default=0)
    question = models.CharField('質問文', max_length=1024)
    answer = models.CharField('回答文', max_length=1024)
    document = models.ForeignKey(Context, verbose_name='ドキュメント', related_name='qacs', on_delete=models.CASCADE, default=0)

    def __str__(self):
        return self.question