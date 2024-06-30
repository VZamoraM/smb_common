echo "Start to connect the SMB"

echo "Listing available Wi-Fi networks:"
nmcli connection show

echo "Connecting to the Wi-Fi network:"
nmcli device wifi connect SMB_263 password SMB_263_RSS

ssh smb-263
