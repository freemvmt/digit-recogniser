adduser --system --group --home /home/mnist --shell /bin/bash mnist
sudo usermod -aG docker mnist
sudo -i -u mnist
