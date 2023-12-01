```
git clone git@github.com:mengwanguc/kubernetes-allox.git
```

## old golang

```
sudo apt remove golang-go
sudo rm -rf /usr/local/go

cd /tmp
wget https://go.dev/dl/go1.10.8.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.10.8.linux-amd64.tar.gz
cd -
```

then add this to `~/.bashrc`

```
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
```

then 

```
source ~/.bashrc
```


## compile
https://github.com/kubernetes/website/blob/release-1.12/content/en/docs/tasks/administer-cluster/configure-multiple-schedulers.md

website/content/en/examples/admin/sched/my-scheduler.yaml

https://stackoverflow.com/questions/37586169/creating-custom-scheduler-doesnt-work

```
cd kubernetes-allox
make

sudo docker build -t wangm12/my-kube-scheduler:v1.12.10 .
sudo docker push wangm12/my-kube-scheduler:v1.12.10
kubectl create -f my-scheduler.yaml
kubectl get pods --namespace=kube-system
kubectl get pods --all-namespaces
```

```
kubectl describe pod my-scheduler-5dbbfd997f-p6bhs -n kube-system
kubectl logs my-scheduler-599f974bd8-sd2hw -n kube-system
```