```
git clone git@github.com:mengwanguc/kubernetes-allox.git
```

## old golang

```
sudo apt remove golang-go

wget https://go.dev/dl/go1.11.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.11.linux-amd64.tar.gz
```

then add this to `~/.bashrc`

```
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
```

## compile
```
cd kubernetes-allox
make

sudo docker build -t wangm12/my-kube-scheduler .
sudo docker push wangm12/my-kube-scheduler
kubectl create -f my-scheduler.yaml
kubectl get pods --namespace=kube-system
```

```
kubectl describe pod my-scheduler-6497bc64c5-vq54z -n kube-system

```