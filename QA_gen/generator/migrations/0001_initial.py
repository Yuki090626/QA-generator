# Generated by Django 3.0.2 on 2021-02-22 05:40

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='QAC',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.CharField(max_length=1024, verbose_name='質問文')),
                ('answer', models.CharField(max_length=1024, verbose_name='回答文')),
                ('context', models.CharField(max_length=1024, verbose_name='ドキュメント')),
            ],
        ),
    ]
