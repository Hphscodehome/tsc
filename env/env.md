背景：电脑A开启代理，电脑B开发代码

# 方法1.电脑B上

`~/.ssh/config`

```bash
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

# 方法2.映射http代理端口

git remote -v 设置为https，即可快速

## 方法2.1，电脑A主动

```bash
RemoteForward 7890 localhost:7890
```

连上后：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

## 方法2.2，电脑B主动

```bash
LocalForward 7890 IP_A:7890
```

连上后：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

问题：电脑A往往没有公网ip。

# 方法3.映射socks代理端口

## 方法3.1，电脑A主动

```bash
RemoteForward 7891 localhost:7891
```

连上后：`~/.ssh/config`

```bash
Host github.com
    HostName github.com
    User git
    ProxyCommand nc -x 127.0.0.1:7890 %h %p
```


## 方法3.2，电脑B主动
```bash
LocalForward 7890 IP_A:7890
```
连上后：
```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```
问题：电脑A往往没有公网ip。