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
make kube-scheduler

sudo docker build -t wangm12/my-kube-scheduler:allox-egpu .
sudo docker push wangm12/my-kube-scheduler:allox-egpu
kubectl delete -f my-scheduler.yaml
kubectl create -f my-scheduler.yaml
kubectl get pods --namespace=kube-system
kubectl get pods --all-namespaces
```

```
kubectl describe pod my-scheduler-5dbbfd997f-p6bhs -n kube-system
kubectl logs my-scheduler-96649cb67-gmtmb -n kube-system
```

Force Kubernetes to pull image: https://www.baeldung.com/ops/kubernetes-pull-image-again#:~:text=One%20way%20to%20force%20Kubernetes,already%20present%20on%20the%20node.


# create namespaces

```
cd pod/sched
bash create-namespaces.sh
kubectl get namespaces --show-labels
```



## Reserve CPU/memory resources



## Notes

scheduler.go: SchedulerOne() ->
factory.go: getNextPod() ->
scheduling_queue.go: PickNextPod(client clientset.Interface) ->
util.go: EqualShare(allPods []*v1.Pod, client clientset.Interface) ->





strDemands := strings.Split(container.Command[cmdIdx], ",")
cpuDemand: strDemands[0]
gpuDemand: strDemands[1]
memDemand: strDemands[2]
cTime: strDemands[3]


AllocatableResource:&cache.Resource{MilliCPU:96000, Memory:201105616896, EphemeralStorage:405676817761, AllowedPodNumber:110, ScalarResources:map[v1.ResourceName]int64{"hugepages-1Gi":0, "hugepages-2Mi":0, "ucare.cs.uchicago.edu/e-gpu":3, "github.com/e-gpu":0}}}