Vagrant.configure("2") do |config|
  config.vm.box = "bento/centos-7"
  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true
  config.vm.provider "virtualbox" do |vb|
    vb.memory = 2048
    vb.cpus = 2
  end
  config.vm.provision "shell", inline: <<-SHELL
    sudo yum install -y opencv opencv-devel opencv-python xauth xorg-x11-xauth gcc-c++ ImageMagick libcanberra-gtk2 PackageKit-gtk3-module
    sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
    sudo yum install -y neovim python36-neovim epel-release snapd
    sudo systemctl enable --now snapd.socket
    sudo ln -s /var/lib/snapd/snap /snap
    # need to perform this step twice for some reason...
    # Think it forces snapd to initialise first time or something idk it works
    sudo snap install clangd --classic
    sudo snap install clangd --classic

    cat >>/home/vagrant/.bashrc <<HERE
      export LD_LIBRARY_PATH=/usr/lib64:
      export CPLUS_INCLUDE_PATH=:/usr/include/opencv2:/usr/include/opencv
      export OPENCV_CFLAGS=-I/usr/include/opencv2
      export O_LIBS="-L/usr/lib64 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann"
HERE
  SHELL

end
