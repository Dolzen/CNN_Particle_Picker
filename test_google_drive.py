from pydrive.auth import GoogleAuth

gauth = GoogleAuth()
# Create local webserver which automatically handles authentication.
gauth.LocalWebserverAuth()

# Create GoogleDrive instance with authenticated GoogleAuth instance.
drive = GoogleDrive(gauth)