# Generated by Django 3.0.2 on 2021-02-24 05:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('generator', '0006_auto_20210224_1429'),
    ]

    operations = [
        migrations.AddField(
            model_name='qac',
            name='document',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.CASCADE, related_name='qacs', to='generator.Context', verbose_name='ドキュメント'),
        ),
    ]
