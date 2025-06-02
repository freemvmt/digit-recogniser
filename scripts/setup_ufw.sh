apt install ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw enable
ufw status verbose

# if we wanted to allow only a specific IP address to access our server on port 8501, we could do something like...
# ufw allow from x.x.x.x/32 to any port 8501 proto tcp
