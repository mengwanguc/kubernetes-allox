type Resource:

kubernetes/pkg/scheduler/framework/types.go

```
// Resource is a collection of compute resource.
type Resource struct {
	MilliCPU         int64
	Memory           int64
	EphemeralStorage int64
	// We store allowedPodNumber (which is Node.Status.Allocatable.Pods().Value())
	// explicitly as int, to avoid conversions and improve performance.
	AllowedPodNumber int
	// ScalarResources
	ScalarResources map[v1.ResourceName]int64
}
```


https://github.com/NVIDIA/k8s-device-plugin/blob/2722aa155ebb77a5edf985018ca3db02cc88eedb/internal/lm/resource.go#L46


