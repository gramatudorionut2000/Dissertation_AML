# users/tests/test_auth.py

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from users.decorators import auditor_required
from django.http import HttpResponse, HttpResponseForbidden

User = get_user_model()


class UserModelTests(TestCase):
    
    def test_create_regular_user(self):
        user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.email, 'test@example.com')
        self.assertFalse(user.is_auditor)
        self.assertTrue(user.check_password('testpass123'))
    
    def test_create_auditor_user(self):
        auditor = User.objects.create_user(
            username='auditor1',
            email='auditor@example.com',
            password='auditorpass123',
            is_auditor=True
        )
        self.assertTrue(auditor.is_auditor)
        self.assertEqual(str(auditor), 'auditor1')


class AuthenticationTests(TestCase):
    
    def setUp(self):
        self.client = Client()
        self.regular_user = User.objects.create_user(
            username='regular',
            password='regularpass123',
            is_auditor=False
        )
        self.auditor_user = User.objects.create_user(
            username='auditor',
            password='auditorpass123',
            is_auditor=True
        )
    
    def test_login_page_accessible(self):
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Log in')
    
    def test_register_page_accessible(self):
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Create a New Account')
    
    def test_successful_login(self):
        response = self.client.post(reverse('login'), {
            'username': 'regular',
            'password': 'regularpass123'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(self.client.session.get('_auth_user_id'))
    
    def test_failed_login(self):
        response = self.client.post(reverse('login'), {
            'username': 'regular',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Invalid username or password')
    
    def test_logout(self):
        self.client.login(username='regular', password='regularpass123')
        response = self.client.post(reverse('logout'))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(self.client.session.get('_auth_user_id'))
    
    def test_profile_requires_login(self):
        response = self.client.get(reverse('profile'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)
    
    def test_profile_accessible_when_logged_in(self):
        self.client.login(username='regular', password='regularpass123')
        response = self.client.get(reverse('profile'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'User Profile')


class AuditorRequiredDecoratorTests(TestCase):
    
    def setUp(self):
        self.client = Client()
        self.regular_user = User.objects.create_user(
            username='regular',
            password='regularpass123',
            is_auditor=False
        )
        self.auditor_user = User.objects.create_user(
            username='auditor',
            password='auditorpass123',
            is_auditor=True
        )
    
    def test_anonymous_user_redirected_to_login(self):
        response = self.client.get(reverse('transactions'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)
    
    def test_regular_user_forbidden(self):
        self.client.login(username='regular', password='regularpass123')
        response = self.client.get(reverse('transactions'))
        self.assertEqual(response.status_code, 403)
        self.assertContains(response, 'You do not have permission', status_code=403)
    
    def test_auditor_user_allowed(self):
        self.client.login(username='auditor', password='auditorpass123')
        response = self.client.get(reverse('transactions'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Transaction Records')


class RegistrationTests(TestCase):
    
    def setUp(self):
        self.client = Client()
    
    def test_register_regular_user(self):
        response = self.client.post(reverse('register'), {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'first_name': 'New',
            'last_name': 'User',
            'password1': 'ComplexPass123!',
            'password2': 'ComplexPass123!',
            'is_auditor': False
        })
        self.assertEqual(response.status_code, 302) 

        user = User.objects.get(username='newuser')
        self.assertFalse(user.is_auditor)
        self.assertEqual(user.email, 'newuser@example.com')
    
    def test_register_auditor_user(self):
        response = self.client.post(reverse('register'), {
            'username': 'newauditor',
            'email': 'auditor@example.com',
            'first_name': 'New',
            'last_name': 'Auditor',
            'password1': 'ComplexPass123!',
            'password2': 'ComplexPass123!',
            'is_auditor': True
        })
        self.assertEqual(response.status_code, 302)
        
        user = User.objects.get(username='newauditor')
        self.assertTrue(user.is_auditor)
    
    def test_register_password_mismatch(self):
        response = self.client.post(reverse('register'), {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'first_name': 'New',
            'last_name': 'User',
            'password1': 'ComplexPass123!',
            'password2': 'DifferentPass123!',
            'is_auditor': False
        })
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, 'password')
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertTrue(form.errors)
        self.assertIn('password2', form.errors)


class FraudDetectorAccessTests(TestCase):
    
    def setUp(self):
        self.client = Client()
        self.regular_user = User.objects.create_user(
            username='regular',
            password='regularpass123',
            is_auditor=False
        )
        self.auditor_user = User.objects.create_user(
            username='auditor',
            password='auditorpass123',
            is_auditor=True
        )
    
    def test_transactions_view_auditor_only(self):
        response = self.client.get(reverse('transactions'))
        self.assertEqual(response.status_code, 302)
        
        self.client.login(username='regular', password='regularpass123')
        response = self.client.get(reverse('transactions'))
        self.assertEqual(response.status_code, 403)
        
        self.client.logout()
        self.client.login(username='auditor', password='auditorpass123')
        response = self.client.get(reverse('transactions'))
        self.assertEqual(response.status_code, 200)
    
    def test_training_view_auditor_only(self):
        self.client.login(username='regular', password='regularpass123')
        response = self.client.get(reverse('training'))
        self.assertEqual(response.status_code, 403)
        
        self.client.logout()
        self.client.login(username='auditor', password='auditorpass123')
        response = self.client.get(reverse('training'))
        self.assertEqual(response.status_code, 200)
    
    def test_inference_view_auditor_only(self):
        self.client.login(username='regular', password='regularpass123')
        response = self.client.get(reverse('inference'))
        self.assertEqual(response.status_code, 403)
        
        self.client.logout()
        self.client.login(username='auditor', password='auditorpass123')
        response = self.client.get(reverse('inference'))
        self.assertEqual(response.status_code, 200)