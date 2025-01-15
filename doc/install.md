# 红米AC2100刷系统
参考：https://www.right.com.cn/forum/thread-4054150-1-1.html
## 刷入breed系统
1.准备低版本固件到电脑上：http://cdn.cnbj1.fds.api.mi-img.com/xiaoqiang/rom/rm2100/miwifi_rm2100_firmware_d6234_2.0.7.bin
2.电脑通过浏览器192.168.31.1进入路由器管理界面，上传本固件，升级（无需网线连接）。
3.升级完成后设置禁止自动升级。
4.电脑通过浏览器192.168.31.1进入路由器管理界面，通过浏览器的地址栏链接获取STOK，格式为：http://192.168.31.1/cgi-bin/luci/;stok=<STOK>/web/home#router
5.利用上面获取的STOK，在电脑浏览器中输入：http://192.168.31.1/cgi-bin/luci/;stok=<STOK>/api/misystem/set_config_iotdev?bssid=Xiaomi&user_id=longdike&ssid=-h%3B%20nvram%20set%20ssh_en%3D1%3B%20nvram%20commit%3B%20sed%20-i%20's%2Fchannel%3D.*%2Fchannel%3D%5C%22debug%5C%22%2Fg'%20%2Fetc%2Finit.d%2Fdropbear%3B%20%2Fetc%2Finit.d%2Fdropbear%20start%3B
6.如果返回401错误，原因可能是版本不正确或者<STOK>值错误或者链接输入不完整等，提示404错误，说明输入地址错误，请检查固件版本或链接地址...
7.建议一键注入后需等待一些时间，保证路由器后台能正确处理注入信息后再重启
8.修改管理员root密码：http://192.168.31.1/cgi-bin/luci/;stok=<STOK>/api/misystem/set_config_iotdev?bssid=Xiaomi&user_id=longdike&ssid=-h%3B%20echo%20-e%20'admin%5Cnadmin'%20%7C%20passwd%20root%3B



12.下载breed固件到电脑上：https://breed.hackpascal.net/breed-mt7621-xiaomi-r3g.bin
13.安装固件

## 刷入openwrt系统

路由器断电--压住reset按钮--路由器插电---等待20s到蓝灯闪烁或常亮---电脑浏览器输入192.168.1.1进入breed后台

选择openwrt中间态固件--更新
然后再安装openwrt小米中文版固件。