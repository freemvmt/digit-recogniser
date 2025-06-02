# Server setup

These scripts are guides to setting up the Hetzner box in anticipation of deploying the Docker triplet there.

They all assume you're already ssh'ed into server as root.

Note that `setup_user.sh` should be run last, since it logs you in as the new `mnist` user (although we can return to the root shell by simply writing `exit`).
