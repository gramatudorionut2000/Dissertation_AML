from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):

    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', 'is_auditor')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'class': 'form-control',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            })
        
        self.fields['is_auditor'].label = "Register as Auditor"
        self.fields['is_auditor'].help_text = "Auditors can review transactions."
        self.fields['password1'].help_text = "Your password must contain at least 8 characters and can't be entirely numeric."
        self.fields['password2'].help_text = "Passwords must match."


    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_auditor = self.cleaned_data['is_auditor']
        if commit:
            user.save()
        return user

class CustomAuthenticationForm(AuthenticationForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'class': 'form-control',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            })

class ProfileUpdateForm(forms.ModelForm):

    class Meta:
        model = CustomUser
        fields = ['first_name', 'last_name', 'email']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'class': 'form-control',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            })

class CustomPasswordChangeForm(PasswordChangeForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'class': 'form-control',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            })