from django.http import HttpResponseForbidden
from functools import wraps

def auditor_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user = request.user
        if not user.is_authenticated:
            from django.contrib.auth.views import redirect_to_login
            return redirect_to_login(request.get_full_path())
        
        if not getattr(user, 'is_auditor', False):
            return HttpResponseForbidden("You do not have permission to access this page.")
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view
