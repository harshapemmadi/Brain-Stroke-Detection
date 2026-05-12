from django.db import models


class UserProfile(models.Model):
    username  = models.CharField(max_length=150, unique=True)
    password  = models.CharField(max_length=255)  # hashed in production
    email     = models.EmailField(unique=True)
    contact   = models.CharField(max_length=20)
    address   = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username


class SegmentationResult(models.Model):
    METHOD_CHOICES = [('OTSU', 'OTSU'), ('DBIM', 'DBIM')]

    user        = models.ForeignKey(UserProfile, on_delete=models.CASCADE,
                                    null=True, blank=True)
    method      = models.CharField(max_length=10, choices=METHOD_CHOICES)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    accuracy    = models.FloatField(default=0.0)
    probability = models.FloatField(default=0.0)
    has_tumor   = models.BooleanField(default=False)
    otsu_thresh = models.FloatField(default=0.0, null=True, blank=True)
    improvement = models.FloatField(default=0.0, null=True, blank=True)
    image_name  = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.method} – {self.accuracy:.2%} – {self.uploaded_at:%Y-%m-%d %H:%M}"
