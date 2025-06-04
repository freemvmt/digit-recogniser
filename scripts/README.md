# Server setup

These scripts are guides to setting up the Hetzner box in anticipation of deploying the Docker stack there. Currently this is a manual job, but the scripts could be polished and consolidated into one _superscript_ to automate the process.

They all assume you're already ssh'ed into server as root.

Note that the commands in `setup_user.sh` should be run last, since they log you in as the `mnist` user, which either might not yet exist, or will not be able to execute most commands from the other scripts. Of course, you can return to the root shell by simply writing `exit`.
