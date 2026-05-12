from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=150, unique=True)),
                ('password', models.CharField(max_length=255)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('contact', models.CharField(max_length=20)),
                ('address', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='SegmentationResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('method', models.CharField(choices=[('OTSU', 'OTSU'), ('DBIM', 'DBIM')], max_length=10)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('accuracy', models.FloatField(default=0.0)),
                ('probability', models.FloatField(default=0.0)),
                ('has_tumor', models.BooleanField(default=False)),
                ('otsu_thresh', models.FloatField(blank=True, default=0.0, null=True)),
                ('improvement', models.FloatField(blank=True, default=0.0, null=True)),
                ('image_name', models.CharField(blank=True, max_length=255)),
                ('user', models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.CASCADE,
                    to='TumorApp.userprofile',
                )),
            ],
        ),
    ]
