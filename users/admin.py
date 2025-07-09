from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff', 'is_auditor', ]
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('is_auditor',)}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ('is_auditor',)}),
    )

admin.site.register(CustomUser, CustomUserAdmin)