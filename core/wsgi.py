"""
WSGI config for core project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os
import site
import sys

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
site.addsitedir('/var/www/kg/venv/lib/python3.6/site-packages')

sys.path.append('/var/www')
sys.path.append('/var/www/kg')

application = get_wsgi_application()
