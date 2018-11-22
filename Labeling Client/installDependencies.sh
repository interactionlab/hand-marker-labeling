##############################################
## REQIURES HOST ONLY NETWORK ADAPTER IN VM ##
##############################################

sudo apt update
sudo apt-get install python3-numpy
sudo python3 -m pip install tensorflow
sudo apt install net-tools
sudo ifconfig enp0s3 multicast
sudo sysctl -w "net.ipv4.conf.all.rp_filter=0"
sudo ip maddr add 239.255.42.99 dev enp0s3
sudo ip addr add 239.255.42.99 dev enp0s3 autojoin
sudo iptables -A INPUT -p udp --dport 1510 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 1511 -j ACCEPT
sudo iptables -A INPUT -m pkttype --pkt-type multicast -j ACCEPT


sudo ifconfig enp0s8 multicast
sudo ip maddr add 239.255.42.99 dev enp0s8
sudo ip addr add 239.255.42.99 dev enp0s8 autojoin
