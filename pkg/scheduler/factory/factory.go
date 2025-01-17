/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package factory can set up a scheduler. This code is here instead of
// cmd/scheduler for both testability and reuse.
package factory

import (
	"fmt"
	"os"
	"os/signal"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/api/policy/v1beta1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	policyinformers "k8s.io/client-go/informers/policy/v1beta1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1beta1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/api/validation"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/core/equivalence"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

const (
	initialGetBackoff = 100 * time.Millisecond
	maximalGetBackoff = time.Minute
)

var (
	serviceAffinitySet            = sets.NewString(predicates.CheckServiceAffinityPred)
	matchInterPodAffinitySet      = sets.NewString(predicates.MatchInterPodAffinityPred)
	generalPredicatesSets         = sets.NewString(predicates.GeneralPred)
	noDiskConflictSet             = sets.NewString(predicates.NoDiskConflictPred)
	maxPDVolumeCountPredicateKeys = []string{predicates.MaxGCEPDVolumeCountPred, predicates.MaxAzureDiskVolumeCountPred, predicates.MaxEBSVolumeCountPred}
	podStartTimes                 = make(map[string]time.Time)     //tanle
	podStopTimes                  = make(map[string]time.Duration) //tanle
	jcts						  = []time.Duration{}
	total_jct 					  time.Duration

)

// configFactory is the default implementation of the scheduler.Configurator interface.
type configFactory struct {
	client clientset.Interface
	// queue for pods that need scheduling
	podQueue core.SchedulingQueue
	// a means to list all known scheduled pods.
	scheduledPodLister corelisters.PodLister
	// a means to list all known scheduled pods and pods assumed to have been scheduled.
	podLister algorithm.PodLister
	// a means to list all nodes
	nodeLister corelisters.NodeLister
	// a means to list all PersistentVolumes
	pVLister corelisters.PersistentVolumeLister
	// a means to list all PersistentVolumeClaims
	pVCLister corelisters.PersistentVolumeClaimLister
	// a means to list all services
	serviceLister corelisters.ServiceLister
	// a means to list all controllers
	controllerLister corelisters.ReplicationControllerLister
	// a means to list all replicasets
	replicaSetLister appslisters.ReplicaSetLister
	// a means to list all statefulsets
	statefulSetLister appslisters.StatefulSetLister
	// a means to list all PodDisruptionBudgets
	pdbLister policylisters.PodDisruptionBudgetLister
	// a means to list all StorageClasses
	storageClassLister storagelisters.StorageClassLister

	// Close this to stop all reflectors
	StopEverything chan struct{}

	scheduledPodsHasSynced cache.InformerSynced

	schedulerCache schedulercache.Cache

	// SchedulerName of a scheduler is used to select which pods will be
	// processed by this scheduler, based on pods's "spec.schedulerName".
	schedulerName string

	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	hardPodAffinitySymmetricWeight int32

	// Equivalence class cache
	equivalencePodCache *equivalence.Cache

	// Enable equivalence class cache
	enableEquivalenceClassCache bool

	// Handles volume binding decisions
	volumeBinder *volumebinder.VolumeBinder

	// always check all predicates even if the middle of one predicate fails.
	alwaysCheckAllPredicates bool

	// Disable pod preemption or not.
	disablePreemption bool

	// percentageOfNodesToScore specifies percentage of all nodes to score in each scheduling cycle.
	percentageOfNodesToScore int32
}

const ReplicateJobFeature = false //tanle
const UpdateJobFeature = false    //tanle
var LogComplTimeFeature = true    //tanle

// ConfigFactoryArgs is a set arguments passed to NewConfigFactory.
type ConfigFactoryArgs struct {
	SchedulerName                  string
	Client                         clientset.Interface
	NodeInformer                   coreinformers.NodeInformer
	PodInformer                    coreinformers.PodInformer
	PvInformer                     coreinformers.PersistentVolumeInformer
	PvcInformer                    coreinformers.PersistentVolumeClaimInformer
	ReplicationControllerInformer  coreinformers.ReplicationControllerInformer
	ReplicaSetInformer             appsinformers.ReplicaSetInformer
	StatefulSetInformer            appsinformers.StatefulSetInformer
	ServiceInformer                coreinformers.ServiceInformer
	PdbInformer                    policyinformers.PodDisruptionBudgetInformer
	StorageClassInformer           storageinformers.StorageClassInformer
	HardPodAffinitySymmetricWeight int32
	EnableEquivalenceClassCache    bool
	DisablePreemption              bool
	PercentageOfNodesToScore       int32
	BindTimeoutSeconds             int64
}

// NewConfigFactory initializes the default implementation of a Configurator To encourage eventual privatization of the struct type, we only
// return the interface.
func NewConfigFactory(args *ConfigFactoryArgs) scheduler.Configurator {
	stopEverything := make(chan struct{})
	schedulerCache := schedulercache.New(30*time.Second, stopEverything)

	// storageClassInformer is only enabled through VolumeScheduling feature gate
	var storageClassLister storagelisters.StorageClassLister
	if args.StorageClassInformer != nil {
		storageClassLister = args.StorageClassInformer.Lister()
	}
	c := &configFactory{
		client:                         args.Client,
		podLister:                      schedulerCache,
		podQueue:                       core.NewSchedulingQueue(),
		pVLister:                       args.PvInformer.Lister(),
		pVCLister:                      args.PvcInformer.Lister(),
		serviceLister:                  args.ServiceInformer.Lister(),
		controllerLister:               args.ReplicationControllerInformer.Lister(),
		replicaSetLister:               args.ReplicaSetInformer.Lister(),
		statefulSetLister:              args.StatefulSetInformer.Lister(),
		pdbLister:                      args.PdbInformer.Lister(),
		storageClassLister:             storageClassLister,
		schedulerCache:                 schedulerCache,
		StopEverything:                 stopEverything,
		schedulerName:                  args.SchedulerName,
		hardPodAffinitySymmetricWeight: args.HardPodAffinitySymmetricWeight,
		enableEquivalenceClassCache:    args.EnableEquivalenceClassCache,
		disablePreemption:              args.DisablePreemption,
		percentageOfNodesToScore:       args.PercentageOfNodesToScore,
	}

	c.scheduledPodsHasSynced = args.PodInformer.Informer().HasSynced
	// scheduled pod cache
	args.PodInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return assignedNonTerminatedPod(t)
				case cache.DeletedFinalStateUnknown:
					if pod, ok := t.Obj.(*v1.Pod); ok {
						return assignedNonTerminatedPod(pod)
					}
					runtime.HandleError(fmt.Errorf("unable to convert object %T to *v1.Pod in %T", obj, c))
					return false
				default:
					runtime.HandleError(fmt.Errorf("unable to handle object in %T: %T", c, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    c.addPodToCache,
				UpdateFunc: c.updatePodInCache,
				DeleteFunc: c.deletePodFromCache,
			},
		},
	)
	// unscheduled pod queue
	args.PodInformer.Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return unassignedNonTerminatedPod(t) && responsibleForPod(t, args.SchedulerName)
				case cache.DeletedFinalStateUnknown:
					if pod, ok := t.Obj.(*v1.Pod); ok {
						return unassignedNonTerminatedPod(pod) && responsibleForPod(pod, args.SchedulerName)
					}
					runtime.HandleError(fmt.Errorf("unable to convert object %T to *v1.Pod in %T", obj, c))
					return false
				default:
					runtime.HandleError(fmt.Errorf("unable to handle object in %T: %T", c, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    c.addPodToSchedulingQueue,
				UpdateFunc: c.updatePodInSchedulingQueue,
				DeleteFunc: c.deletePodFromSchedulingQueue,
			},
		},
	)
	// ScheduledPodLister is something we provide to plug-in functions that
	// they may need to call.
	c.scheduledPodLister = assignedPodLister{args.PodInformer.Lister()}

	args.NodeInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.addNodeToCache,
			UpdateFunc: c.updateNodeInCache,
			DeleteFunc: c.deleteNodeFromCache,
		},
	)
	c.nodeLister = args.NodeInformer.Lister()

	args.PdbInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.addPDBToCache,
			UpdateFunc: c.updatePDBInCache,
			DeleteFunc: c.deletePDBFromCache,
		},
	)
	c.pdbLister = args.PdbInformer.Lister()

	// On add and delete of PVs, it will affect equivalence cache items
	// related to persistent volume
	args.PvInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			// MaxPDVolumeCountPredicate: since it relies on the counts of PV.
			AddFunc:    c.onPvAdd,
			UpdateFunc: c.onPvUpdate,
			DeleteFunc: c.onPvDelete,
		},
	)
	c.pVLister = args.PvInformer.Lister()

	// This is for MaxPDVolumeCountPredicate: add/delete PVC will affect counts of PV when it is bound.
	args.PvcInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.onPvcAdd,
			UpdateFunc: c.onPvcUpdate,
			DeleteFunc: c.onPvcDelete,
		},
	)
	c.pVCLister = args.PvcInformer.Lister()

	// This is for ServiceAffinity: affected by the selector of the service is updated.
	// Also, if new service is added, equivalence cache will also become invalid since
	// existing pods may be "captured" by this service and change this predicate result.
	args.ServiceInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.onServiceAdd,
			UpdateFunc: c.onServiceUpdate,
			DeleteFunc: c.onServiceDelete,
		},
	)
	c.serviceLister = args.ServiceInformer.Lister()

	// Existing equivalence cache should not be affected by add/delete RC/Deployment etc,
	// it only make sense when pod is scheduled or deleted

	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		// Setup volume binder
		c.volumeBinder = volumebinder.NewVolumeBinder(args.Client, args.PvcInformer, args.PvInformer, args.StorageClassInformer, time.Duration(args.BindTimeoutSeconds)*time.Second)

		args.StorageClassInformer.Informer().AddEventHandler(
			cache.ResourceEventHandlerFuncs{
				AddFunc:    c.onStorageClassAdd,
				DeleteFunc: c.onStorageClassDelete,
			},
		)
	}

	// Setup cache comparer
	comparer := &cacheComparer{
		podLister:  args.PodInformer.Lister(),
		nodeLister: args.NodeInformer.Lister(),
		pdbLister:  args.PdbInformer.Lister(),
		cache:      c.schedulerCache,
		podQueue:   c.podQueue,
	}

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, compareSignal)

	go func() {
		for {
			select {
			case <-c.StopEverything:
				return
			case <-ch:
				comparer.Compare()
			}
		}
	}()

	return c
}

// skipPodUpdate checks whether the specified pod update should be ignored.
// This function will return true if
//   - The pod has already been assumed, AND
//   - The pod has only its ResourceVersion, Spec.NodeName and/or Annotations
//     updated.
func (c *configFactory) skipPodUpdate(pod *v1.Pod) bool {
	// Non-assumed pods should never be skipped.
	isAssumed, err := c.schedulerCache.IsAssumedPod(pod)
	if err != nil {
		runtime.HandleError(fmt.Errorf("failed to check whether pod %s/%s is assumed: %v", pod.Namespace, pod.Name, err))
		return false
	}
	if !isAssumed {
		return false
	}

	// Gets the assumed pod from the cache.
	assumedPod, err := c.schedulerCache.GetPod(pod)
	if err != nil {
		runtime.HandleError(fmt.Errorf("failed to get assumed pod %s/%s from cache: %v", pod.Namespace, pod.Name, err))
		return false
	}

	// Compares the assumed pod in the cache with the pod update. If they are
	// equal (with certain fields excluded), this pod update will be skipped.
	f := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		// ResourceVersion must be excluded because each object update will
		// have a new resource version.
		p.ResourceVersion = ""
		// Spec.NodeName must be excluded because the pod assumed in the cache
		// is expected to have a node assigned while the pod update may nor may
		// not have this field set.
		p.Spec.NodeName = ""
		// Annotations must be excluded for the reasons described in
		// https://github.com/kubernetes/kubernetes/issues/52914.
		p.Annotations = nil
		return p
	}
	assumedPodCopy, podCopy := f(assumedPod), f(pod)
	if !reflect.DeepEqual(assumedPodCopy, podCopy) {
		return false
	}
	glog.V(3).Infof("Skipping pod %s/%s update", pod.Namespace, pod.Name)
	return true
}

func (c *configFactory) onPvAdd(obj interface{}) {
	if c.enableEquivalenceClassCache {
		pv, ok := obj.(*v1.PersistentVolume)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolume: %v", obj)
			return
		}
		c.invalidatePredicatesForPv(pv)
	}
	// Pods created when there are no PVs available will be stuck in
	// unschedulable queue. But unbound PVs created for static provisioning and
	// delay binding storage class are skipped in PV controller dynamic
	// provisiong and binding process, will not trigger events to schedule pod
	// again. So we need to move pods to active queue on PV add for this
	// scenario.
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) onPvUpdate(old, new interface{}) {
	if c.enableEquivalenceClassCache {
		newPV, ok := new.(*v1.PersistentVolume)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolume: %v", new)
			return
		}
		oldPV, ok := old.(*v1.PersistentVolume)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolume: %v", old)
			return
		}
		c.invalidatePredicatesForPvUpdate(oldPV, newPV)
	}
	// Scheduler.bindVolumesWorker may fail to update assumed pod volume
	// bindings due to conflicts if PVs are updated by PV controller or other
	// parties, then scheduler will add pod back to unschedulable queue. We
	// need to move pods to active queue on PV update for this scenario.
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) invalidatePredicatesForPvUpdate(oldPV, newPV *v1.PersistentVolume) {
	invalidPredicates := sets.NewString()
	// CheckVolumeBinding predicate calls SchedulerVolumeBinder.FindPodVolumes
	// which will cache PVs in PodBindingCache. When PV got updated, we should
	// invalidate cache, otherwise PVAssumeCache.Assume will fail with out of sync
	// error.
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		invalidPredicates.Insert(predicates.CheckVolumeBindingPred)
	}
	for k, v := range newPV.Labels {
		// If PV update modifies the zone/region labels.
		if isZoneRegionLabel(k) && !reflect.DeepEqual(v, oldPV.Labels[k]) {
			invalidPredicates.Insert(predicates.NoVolumeZoneConflictPred)
			break
		}
	}
	c.equivalencePodCache.InvalidatePredicates(invalidPredicates)
}

// isZoneRegionLabel check if given key of label is zone or region label.
func isZoneRegionLabel(k string) bool {
	return k == kubeletapis.LabelZoneFailureDomain || k == kubeletapis.LabelZoneRegion
}

func (c *configFactory) onPvDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		var pv *v1.PersistentVolume
		switch t := obj.(type) {
		case *v1.PersistentVolume:
			pv = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			pv, ok = t.Obj.(*v1.PersistentVolume)
			if !ok {
				glog.Errorf("cannot convert to *v1.PersistentVolume: %v", t.Obj)
				return
			}
		default:
			glog.Errorf("cannot convert to *v1.PersistentVolume: %v", t)
			return
		}
		c.invalidatePredicatesForPv(pv)
	}
}

func (c *configFactory) invalidatePredicatesForPv(pv *v1.PersistentVolume) {
	// You could have a PVC that points to a PV, but the PV object doesn't exist.
	// So when the PV object gets added, we can recount.
	invalidPredicates := sets.NewString()

	// PV types which impact MaxPDVolumeCountPredicate
	if pv.Spec.AWSElasticBlockStore != nil {
		invalidPredicates.Insert(predicates.MaxEBSVolumeCountPred)
	}
	if pv.Spec.GCEPersistentDisk != nil {
		invalidPredicates.Insert(predicates.MaxGCEPDVolumeCountPred)
	}
	if pv.Spec.AzureDisk != nil {
		invalidPredicates.Insert(predicates.MaxAzureDiskVolumeCountPred)
	}

	if pv.Spec.CSI != nil && utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		invalidPredicates.Insert(predicates.MaxCSIVolumeCountPred)
	}

	// If PV contains zone related label, it may impact cached NoVolumeZoneConflict
	for k := range pv.Labels {
		if isZoneRegionLabel(k) {
			invalidPredicates.Insert(predicates.NoVolumeZoneConflictPred)
			break
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		// Add/delete impacts the available PVs to choose from
		invalidPredicates.Insert(predicates.CheckVolumeBindingPred)
	}

	c.equivalencePodCache.InvalidatePredicates(invalidPredicates)
}

func (c *configFactory) onPvcAdd(obj interface{}) {
	if c.enableEquivalenceClassCache {
		pvc, ok := obj.(*v1.PersistentVolumeClaim)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", obj)
			return
		}
		c.invalidatePredicatesForPvc(pvc)
	}
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) onPvcUpdate(old, new interface{}) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		return
	}

	if c.enableEquivalenceClassCache {
		newPVC, ok := new.(*v1.PersistentVolumeClaim)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", new)
			return
		}
		oldPVC, ok := old.(*v1.PersistentVolumeClaim)
		if !ok {
			glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", old)
			return
		}
		c.invalidatePredicatesForPvcUpdate(oldPVC, newPVC)
	}
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) onPvcDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		var pvc *v1.PersistentVolumeClaim
		switch t := obj.(type) {
		case *v1.PersistentVolumeClaim:
			pvc = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			pvc, ok = t.Obj.(*v1.PersistentVolumeClaim)
			if !ok {
				glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", t.Obj)
				return
			}
		default:
			glog.Errorf("cannot convert to *v1.PersistentVolumeClaim: %v", t)
			return
		}
		c.invalidatePredicatesForPvc(pvc)
	}
}

func (c *configFactory) invalidatePredicatesForPvc(pvc *v1.PersistentVolumeClaim) {
	// We need to do this here because the ecache uses PVC uid as part of equivalence hash of pod

	// The bound volume type may change
	invalidPredicates := sets.NewString(maxPDVolumeCountPredicateKeys...)

	if utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		invalidPredicates.Insert(predicates.MaxCSIVolumeCountPred)
	}

	// The bound volume's label may change
	invalidPredicates.Insert(predicates.NoVolumeZoneConflictPred)

	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		// Add/delete impacts the available PVs to choose from
		invalidPredicates.Insert(predicates.CheckVolumeBindingPred)
	}
	c.equivalencePodCache.InvalidatePredicates(invalidPredicates)
}

func (c *configFactory) invalidatePredicatesForPvcUpdate(old, new *v1.PersistentVolumeClaim) {
	invalidPredicates := sets.NewString()

	if old.Spec.VolumeName != new.Spec.VolumeName {
		if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
			// PVC volume binding has changed
			invalidPredicates.Insert(predicates.CheckVolumeBindingPred)
		}
		// The bound volume type may change
		invalidPredicates.Insert(maxPDVolumeCountPredicateKeys...)

		if utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
			invalidPredicates.Insert(predicates.MaxCSIVolumeCountPred)
		}
	}

	c.equivalencePodCache.InvalidatePredicates(invalidPredicates)
}

func (c *configFactory) onStorageClassAdd(obj interface{}) {
	sc, ok := obj.(*storagev1.StorageClass)
	if !ok {
		glog.Errorf("cannot convert to *storagev1.StorageClass: %v", obj)
		return
	}

	// CheckVolumeBindingPred fails if pod has unbound immediate PVCs. If these
	// PVCs have specified StorageClass name, creating StorageClass objects
	// with late binding will cause predicates to pass, so we need to move pods
	// to active queue.
	// We don't need to invalidate cached results because results will not be
	// cached for pod that has unbound immediate PVCs.
	if sc.VolumeBindingMode != nil && *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
		c.podQueue.MoveAllToActiveQueue()
	}
}

func (c *configFactory) onStorageClassDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		var sc *storagev1.StorageClass
		switch t := obj.(type) {
		case *storagev1.StorageClass:
			sc = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			sc, ok = t.Obj.(*storagev1.StorageClass)
			if !ok {
				glog.Errorf("cannot convert to *storagev1.StorageClass: %v", t.Obj)
				return
			}
		default:
			glog.Errorf("cannot convert to *storagev1.StorageClass: %v", t)
			return
		}
		c.invalidatePredicatesForStorageClass(sc)
	}
}

func (c *configFactory) invalidatePredicatesForStorageClass(sc *storagev1.StorageClass) {
	invalidPredicates := sets.NewString()

	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		if sc.VolumeBindingMode != nil && *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
			// Delete can cause predicates to fail
			invalidPredicates.Insert(predicates.CheckVolumeBindingPred)
			invalidPredicates.Insert(predicates.NoVolumeZoneConflictPred)
		}
	}

	c.equivalencePodCache.InvalidatePredicates(invalidPredicates)
}

func (c *configFactory) onServiceAdd(obj interface{}) {
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache.InvalidatePredicates(serviceAffinitySet)
	}
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) onServiceUpdate(oldObj interface{}, newObj interface{}) {
	if c.enableEquivalenceClassCache {
		// TODO(resouer) We may need to invalidate this for specified group of pods only
		oldService := oldObj.(*v1.Service)
		newService := newObj.(*v1.Service)
		if !reflect.DeepEqual(oldService.Spec.Selector, newService.Spec.Selector) {
			c.equivalencePodCache.InvalidatePredicates(serviceAffinitySet)
		}
	}
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) onServiceDelete(obj interface{}) {
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache.InvalidatePredicates(serviceAffinitySet)
	}
	c.podQueue.MoveAllToActiveQueue()
}

// GetNodeStore provides the cache to the nodes, mostly internal use, but may also be called by mock-tests.
func (c *configFactory) GetNodeLister() corelisters.NodeLister {
	return c.nodeLister
}

func (c *configFactory) GetHardPodAffinitySymmetricWeight() int32 {
	return c.hardPodAffinitySymmetricWeight
}

func (c *configFactory) GetSchedulerName() string {
	return c.schedulerName
}

// GetClient provides a kubernetes Client, mostly internal use, but may also be called by mock-tests.
func (c *configFactory) GetClient() clientset.Interface {
	return c.client
}

// GetScheduledPodListerIndexer provides a pod lister, mostly internal use, but may also be called by mock-tests.
func (c *configFactory) GetScheduledPodLister() corelisters.PodLister {
	return c.scheduledPodLister
}

func (c *configFactory) addPodToCache(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		glog.Errorf("cannot convert to *v1.Pod: %v", obj)
		return
	}

	if err := c.schedulerCache.AddPod(pod); err != nil {
		glog.Errorf("scheduler cache AddPod failed: %v", err)
	}

	c.podQueue.AssignedPodAdded(pod)

	// NOTE: Updating equivalence cache of addPodToCache has been
	// handled optimistically in: pkg/scheduler/scheduler.go#assume()
}

// update fair score
func (f *configFactory) UpdateFairScore() {
	podClient := f.client.CoreV1().Pods(v1.NamespaceAll)
	pods, err := podClient.List(metav1.ListOptions{
		LabelSelector: labels.Everything().String(),
	})
	if err == nil {
		for _, pod := range pods.Items {
			if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
				schedulercache.UpdateFairScore(&pod, false)
			}
		}
	} else {
		glog.Errorf("[tanle] error %v", err)
	}
}

// tanle delete pod from scheduling queue
func (factory *configFactory) DeletePod(podQueue core.SchedulingQueue) func(pod *v1.Pod) {
	return func(pod *v1.Pod) {
		podQueue.Delete(pod)
	}
}

func (c *configFactory) updatePodInCache(oldObj, newObj interface{}) {
	oldPod, ok := oldObj.(*v1.Pod)

	//tanle list all pods whenever there is an update.
	// glog.Infof("[tanle] updatePodInCache")
	if LogComplTimeFeature {
		c.computeComplTimeForAllPods(false)
	}
	// c.UpdateFairScore()

	if !ok {
		glog.Errorf("cannot convert oldObj to *v1.Pod: %v", oldObj)
		return
	}
	newPod, ok := newObj.(*v1.Pod)
	if !ok {
		glog.Errorf("cannot convert newObj to *v1.Pod: %v", newObj)
		return
	}

	// NOTE: Because the scheduler uses snapshots of schedulerCache and the live
	// version of equivalencePodCache, updates must be written to schedulerCache
	// before invalidating equivalencePodCache.
	if err := c.schedulerCache.UpdatePod(oldPod, newPod); err != nil {
		glog.Errorf("scheduler cache UpdatePod failed: %v", err)
	}

	c.invalidateCachedPredicatesOnUpdatePod(newPod, oldPod)
	c.podQueue.AssignedPodUpdated(newPod)
}

func (f *configFactory) computeComplTimeForAllPods(isCompleted bool) {
	// glog.Infof("[tanle] list all completed pods")
	podClient := f.client.CoreV1().Pods(v1.NamespaceAll)
	pods, err := podClient.List(metav1.ListOptions{
		LabelSelector: labels.Everything().String(),
	})
	if err == nil {
		if isCompleted {
			glog.Infof("[meng] computeComplTimeForAllPods completed")
		}
		for _, pod := range pods.Items {
			if isCompleted && pod.Status.Phase == v1.PodSucceeded {
				// glog.Infof("[tanle] %v %s", pod.Name, pod.Status.Phase)
				if podStopTimes[pod.Name] == 0 {
					podStopTimes[pod.Name] = time.Since(podStartTimes[pod.Name])
					// fmt.Println(" %v", podStopTimes)
					glog.Infof("[tanle] Event pod completed %v/%s", pod.Namespace, pod.Name, pod.Status.Phase)
					// schedulercache.UpdateFairScore(&pod, false)
					schedulercache.DeletePodFromCache(&pod)
				}
			} else if !isCompleted && pod.Status.Phase == v1.PodRunning && pod.Namespace != "kube-scheduler" {
				// glog.Infof("[tanle] %v %s", pod.Name, pod.Status.Phase)
				if podStartTimes[pod.Name].IsZero() {
					podStartTimes[pod.Name] = time.Now()
				}
			} else if isCompleted {
				if podStopTimes[pod.Name] == 0 && pod.Namespace != "kube-scheduler" && pod.Namespace != "kube-system"{
					// glog.Infof("[meng] Event pod completed podStopTimes = 0 %v/%s, %v", pod.Namespace, pod.Name, pod.Status.Phase)
				}
			}
		}
	} else {
		glog.Errorf("[tanle] error %v", err)
		glog.Infof("[meng] error %v", err)
	}
}

const GI = 1024 * 1024 * 1024

//tanle
func (c *configFactory) replicatePod(pod *v1.Pod, isGPU bool) {
	cpuSuffix := "-rcpu"
	gpuSuffix := "-rgpu"
	if strings.Contains(pod.Name, cpuSuffix) || strings.Contains(pod.Name, gpuSuffix) {
		return
	}
	// replicatedPod := obj.(*v1.Pod).DeepCopy()
	sufix := cpuSuffix
	if isGPU {
		sufix = gpuSuffix
	}

	replicatedPod := pod.DeepCopy()
	replicatedPod.Name = replicatedPod.Name + sufix
	replicatedPod.Namespace = "default"

	for cName, container := range replicatedPod.Spec.Containers {
		container.Name = container.Name + sufix
		isGpuJob := false
		if strings.Contains(container.Image, "gpu") {
			isGpuJob = true
		}
		if isGPU {
			container.Image = "wangm12/gpemu-pytorch:egpu"
		} else {
			container.Image = "wangm12/gpemu-pytorch:cpu"
		}
		mainCmd := container.Command[2]
		secCmd := container.Command[3]
		if isGpuJob != isGPU {
			mainCmd = container.Command[3]
			secCmd = container.Command[2]
		}
		container.Command[2] = mainCmd[0:strings.LastIndex(mainCmd, "--num_batches=")] + "--num_batches=100"
		container.Command[3] = secCmd[0:strings.LastIndex(secCmd, "--num_batches=")] + "--num_batches=100"

		// demands
		// primCpu, primGpu, primMem := GetPrimDemand(pod)
		secCpu, secGpu, secMem := GetSecondaryDemand(pod)

		for rName := range container.Resources.Requests {
			quantity := container.Resources.Requests[rName]
			switch rName {
			case v1.ResourceCPU:
				if isGpuJob != isGPU {
					quantity.SetMilli(secCpu)
				}
			case schedulercache.NvidiaGPU:
				if isGpuJob != isGPU {
					quantity.Set(secGpu)
				}
			case v1.ResourceMemory:
				if isGpuJob != isGPU {
					quantity.Set(secMem * GI)
				}
			}

			container.Resources.Requests[rName] = quantity
			container.Resources.Limits[rName] = quantity
		}
		replicatedPod.Spec.Containers[cName] = container
	}

	replicatedPod.ResourceVersion = ""
	replicatedPod.Spec.NodeName = ""
	replicatedPod.Annotations = nil

	podClient := c.client.CoreV1().Pods(replicatedPod.Namespace)
	if _, err := podClient.Create(replicatedPod); err != nil {
		runtime.HandleError(fmt.Errorf("unable to create pod in kubectl %T: %v", replicatedPod, err))
	}
}

func GetSecondaryDemand(pod *v1.Pod) (int64, int64, int64) {
	milliCPU := int64(0)
	gpu := int64(0)
	memInGi := int64(0)
	for _, container := range pod.Spec.Containers {
		// switch demands
		secDemand := container.Command[5]

		strDemands := strings.Split(secDemand, ",")
		cpuDemand, err := strconv.ParseInt(strDemands[0], 10, 64)
		if err != nil {
			glog.Infof("Failed  to convert cpuDemand %s to int64", strDemands[0])
		}
		gpuDemand, err := strconv.ParseInt(strDemands[1], 10, 64)
		if err != nil {
			glog.Infof("Failed to convert gpuDemand %s to int64", strDemands[1])
		}
		memory, err := strconv.ParseInt(strDemands[2], 10, 64)
		if err != nil {
			glog.Infof("Failed to convert memory %s to int64", strDemands[2])
		}

		milliCPU += cpuDemand
		gpu += gpuDemand
		memInGi += memory
	}

	return milliCPU, gpu, memInGi
}

func GetPrimDemand(pod *v1.Pod) (int64, int64, int64) {
	milliCPU := int64(0)
	gpu := int64(0)
	memInGi := int64(0)
	for _, container := range pod.Spec.Containers {
		// switch demands
		secDemand := container.Command[4]

		strDemands := strings.Split(secDemand, ",")
		cpuDemand, err := strconv.ParseInt(strDemands[0], 10, 64)
		if err != nil {
			glog.Infof("Failed  to convert cpuDemand %s to int64", strDemands[0])
		}
		gpuDemand, err := strconv.ParseInt(strDemands[1], 10, 64)
		if err != nil {
			glog.Infof("Failed to convert gpuDemand %s to int64", strDemands[1])
		}
		memory, err := strconv.ParseInt(strDemands[2], 10, 64)
		if err != nil {
			glog.Infof("Failed to convert memory %s to int64", strDemands[2])
		}

		milliCPU += cpuDemand
		gpu += gpuDemand
		memInGi += memory
	}

	return milliCPU, gpu, memInGi
}

func (c *configFactory) switchPod(pod *v1.Pod) {
	isGpuJob := false
	toBeGPU := false
	// check the device of the pod
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Image, "gpu") {
			isGpuJob = true
			break
		}
	}

	if !isGpuJob {
		return
	}
	// delete the pod in kube system
	replicatedPod := pod.DeepCopy()

	for cName, container := range replicatedPod.Spec.Containers {
		if toBeGPU {
			container.Image = "wangm12/gpemu-pytorch:egpu"
		} else {
			container.Image = "wangm12/gpemu-pytorch:cpu"
		}
		mainCmd := container.Command[3]
		container.Command[3] = container.Command[2]
		container.Command[2] = mainCmd

		for rName := range container.Resources.Requests {
			quantity := container.Resources.Requests[rName]
			switch rName {
			case v1.ResourceCPU:
				if toBeGPU {
					quantity.Set(1)
				} else {
					quantity.Set(16)
				}
			case schedulercache.NvidiaGPU:
				if toBeGPU {
					quantity.Set(1)
				} else {
					quantity.Set(0)
				}
			}
			container.Resources.Requests[rName] = quantity
			container.Resources.Limits[rName] = quantity
		}
		replicatedPod.Spec.Containers[cName] = container
	}

	replicatedPod.ResourceVersion = ""
	replicatedPod.Spec.NodeName = ""
	replicatedPod.Annotations = nil

	// p.Client.CoreV1().Pods(pod.Namespace).Delete(pod.Name, &metav1.DeleteOptions{})
	if err := c.client.CoreV1().Pods(pod.Namespace).Delete(pod.Name, &metav1.DeleteOptions{}); err != nil {
		runtime.HandleError(fmt.Errorf("unable to DELETE pod in kubectl %T: %v", pod, err))
	}
	if _, err := c.client.CoreV1().Pods(replicatedPod.Namespace).Create(replicatedPod); err != nil {
		runtime.HandleError(fmt.Errorf("unable to CREATE pod in kubectl %T: %v", replicatedPod, err))
	}
}

//tanle
func (c *configFactory) updatePod(pod *v1.Pod) {
	glog.Infof("[tanle] starting update pod %s ", pod.Name)
	for cName, container := range pod.Spec.Containers {
		for rName := range container.Resources.Requests {
			quantity := container.Resources.Requests[rName]
			switch rName {
			case v1.ResourceCPU:
				quantity.SetMilli(4000)
				glog.Infof("[tanle] starting update %s cpu to %d ", pod.Name, 4000)
			}

			container.Resources.Requests[rName] = quantity
			container.Resources.Limits[rName] = quantity
		}
		pod.Spec.Containers[cName] = container
	}

	podClient := c.client.CoreV1().Pods(pod.Namespace)
	if _, err := podClient.Update(pod); err != nil {
		runtime.HandleError(fmt.Errorf("unable to Update pod in kubectl %T: %v", pod, err))
	}
}

func (c *configFactory) addPodToSchedulingQueue(obj interface{}) {
	//tanle: create 2 replicate pods that profile CPU & GPU jobs.

	if ReplicateJobFeature {
		// glog.Infof("replicate %s", obj.(*v1.Pod).Name)
		c.replicatePod(obj.(*v1.Pod), false)
		// c.replicatePod(obj.(*v1.Pod), true)
	}

	if UpdateJobFeature {
		c.updatePod(obj.(*v1.Pod))
	}

	if err := c.podQueue.Add(obj.(*v1.Pod)); err != nil {
		runtime.HandleError(fmt.Errorf("unable to queue %T: %v", obj, err))
	} else {
		// tanle
		currTime := schedulercache.GetCurrentTime()
		schedulercache.ArrivalTimes[obj.(*v1.Pod).Name] = currTime
	}
}

func (c *configFactory) updatePodInSchedulingQueue(oldObj, newObj interface{}) {
	pod := newObj.(*v1.Pod)
	if c.skipPodUpdate(pod) {
		return
	}
	if err := c.podQueue.Update(oldObj.(*v1.Pod), pod); err != nil {
		runtime.HandleError(fmt.Errorf("unable to update %T: %v", newObj, err))
	}
}

func (c *configFactory) deletePodFromSchedulingQueue(obj interface{}) {
	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = obj.(*v1.Pod)
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			runtime.HandleError(fmt.Errorf("unable to convert object %T to *v1.Pod in %T", obj, c))
			return
		}
	default:
		runtime.HandleError(fmt.Errorf("unable to handle object in %T: %T", c, obj))
		return
	}
	if err := c.podQueue.Delete(pod); err != nil {
		runtime.HandleError(fmt.Errorf("unable to dequeue %T: %v", obj, err))
	}
	if c.volumeBinder != nil {
		// Volume binder only wants to keep unassigned pods
		c.volumeBinder.DeletePodBindings(pod)
	}
}

func (c *configFactory) invalidateCachedPredicatesOnUpdatePod(newPod *v1.Pod, oldPod *v1.Pod) {
	if c.enableEquivalenceClassCache {
		// if the pod does not have bound node, updating equivalence cache is meaningless;
		// if pod's bound node has been changed, that case should be handled by pod add & delete.
		if len(newPod.Spec.NodeName) != 0 && newPod.Spec.NodeName == oldPod.Spec.NodeName {
			if !reflect.DeepEqual(oldPod.GetLabels(), newPod.GetLabels()) {
				// MatchInterPodAffinity need to be reconsidered for this node,
				// as well as all nodes in its same failure domain.
				c.equivalencePodCache.InvalidatePredicates(
					matchInterPodAffinitySet)
			}
			// if requested container resource changed, invalidate GeneralPredicates of this node
			if !reflect.DeepEqual(predicates.GetResourceRequest(newPod),
				predicates.GetResourceRequest(oldPod)) {
				c.equivalencePodCache.InvalidatePredicatesOnNode(
					newPod.Spec.NodeName, generalPredicatesSets)
			}
		}
	}
}

func (c *configFactory) deletePodFromCache(obj interface{}) {

	glog.Infof("[meng] deletePodFromCache")

	if !schedulercache.SCHEDULE_START.IsZero() {
		jct := time.Since(schedulercache.SCHEDULE_START)
		jcts = append(jcts, jct)
		total_jct += jct
		average_jct := total_jct / time.Duration(len(jcts))
		glog.Infof("[meng] JCT: %v, total jct: %v, count: %v, average JCT: %v", 
					jct, total_jct, len(jcts), average_jct)
	}
	

	if LogComplTimeFeature {
		c.computeComplTimeForAllPods(true)
	}

	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = t
		glog.Infof("[meng] check delete pod %v/%s, %v", pod.Namespace, pod.Name, pod.Status.Phase)
		if pod.Namespace != "kube-scheduler" && pod.Namespace != "kube-system" {
			schedulercache.UpdateFairScore(pod, false)
		}
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			glog.Errorf("cannot convert to *v1.Pod: %v", t.Obj)
			glog.Infof("[meng] t.obj cannot convert to *v1.Pod: %v", t.Obj)
			return
		} else {
			glog.Infof("[meng] check delete pod %v/%s, %v", pod.Namespace, pod.Name, pod.Status.Phase)
			if pod.Namespace != "kube-scheduler" && pod.Namespace != "kube-system" {
				schedulercache.UpdateFairScore(pod, false)
			}
		}
	default:
		glog.Errorf("cannot convert to *v1.Pod: %v", t)
		glog.Infof("[meng] t cannot convert to *v1.Pod: %v", t)
		return
	}
	// NOTE: Because the scheduler uses snapshots of schedulerCache and the live
	// version of equivalencePodCache, updates must be written to schedulerCache
	// before invalidating equivalencePodCache.
	if err := c.schedulerCache.RemovePod(pod); err != nil {
		glog.Errorf("scheduler cache RemovePod failed: %v", err)
	}

	c.invalidateCachedPredicatesOnDeletePod(pod)
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) invalidateCachedPredicatesOnDeletePod(pod *v1.Pod) {
	if c.enableEquivalenceClassCache {
		// part of this case is the same as pod add.
		c.equivalencePodCache.InvalidateCachedPredicateItemForPodAdd(pod, pod.Spec.NodeName)
		// MatchInterPodAffinity need to be reconsidered for this node,
		// as well as all nodes in its same failure domain.
		// TODO(resouer) can we just do this for nodes in the same failure domain
		c.equivalencePodCache.InvalidatePredicates(
			matchInterPodAffinitySet)

		// if this pod have these PV, cached result of disk conflict will become invalid.
		for _, volume := range pod.Spec.Volumes {
			if volume.GCEPersistentDisk != nil || volume.AWSElasticBlockStore != nil ||
				volume.RBD != nil || volume.ISCSI != nil {
				c.equivalencePodCache.InvalidatePredicatesOnNode(
					pod.Spec.NodeName, noDiskConflictSet)
			}
		}
	}
}

func (c *configFactory) addNodeToCache(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		glog.Errorf("cannot convert to *v1.Node: %v", obj)
		return
	}

	if err := c.schedulerCache.AddNode(node); err != nil {
		glog.Errorf("scheduler cache AddNode failed: %v", err)
	}

	if c.enableEquivalenceClassCache {
		// GetNodeCache() will lazily create NodeCache for given node if it does not exist.
		c.equivalencePodCache.GetNodeCache(node.GetName())
	}

	c.podQueue.MoveAllToActiveQueue()
	// NOTE: add a new node does not affect existing predicates in equivalence cache
}

func (c *configFactory) updateNodeInCache(oldObj, newObj interface{}) {
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		glog.Errorf("cannot convert oldObj to *v1.Node: %v", oldObj)
		return
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok {
		glog.Errorf("cannot convert newObj to *v1.Node: %v", newObj)
		return
	}

	// NOTE: Because the scheduler uses snapshots of schedulerCache and the live
	// version of equivalencePodCache, updates must be written to schedulerCache
	// before invalidating equivalencePodCache.
	if err := c.schedulerCache.UpdateNode(oldNode, newNode); err != nil {
		glog.Errorf("scheduler cache UpdateNode failed: %v", err)
	}

	c.invalidateCachedPredicatesOnNodeUpdate(newNode, oldNode)
	c.podQueue.MoveAllToActiveQueue()
}

func (c *configFactory) invalidateCachedPredicatesOnNodeUpdate(newNode *v1.Node, oldNode *v1.Node) {
	if c.enableEquivalenceClassCache {
		// Begin to update equivalence cache based on node update
		// TODO(resouer): think about lazily initialize this set
		invalidPredicates := sets.NewString()

		if !reflect.DeepEqual(oldNode.Status.Allocatable, newNode.Status.Allocatable) {
			invalidPredicates.Insert(predicates.GeneralPred) // "PodFitsResources"
		}
		if !reflect.DeepEqual(oldNode.GetLabels(), newNode.GetLabels()) {
			invalidPredicates.Insert(predicates.GeneralPred, predicates.CheckServiceAffinityPred) // "PodSelectorMatches"
			for k, v := range oldNode.GetLabels() {
				// any label can be topology key of pod, we have to invalidate in all cases
				if v != newNode.GetLabels()[k] {
					invalidPredicates.Insert(predicates.MatchInterPodAffinityPred)
				}
				// NoVolumeZoneConflict will only be affected by zone related label change
				if isZoneRegionLabel(k) {
					if v != newNode.GetLabels()[k] {
						invalidPredicates.Insert(predicates.NoVolumeZoneConflictPred)
					}
				}
			}
		}

		oldTaints, oldErr := helper.GetTaintsFromNodeAnnotations(oldNode.GetAnnotations())
		if oldErr != nil {
			glog.Errorf("Failed to get taints from old node annotation for equivalence cache")
		}
		newTaints, newErr := helper.GetTaintsFromNodeAnnotations(newNode.GetAnnotations())
		if newErr != nil {
			glog.Errorf("Failed to get taints from new node annotation for equivalence cache")
		}
		if !reflect.DeepEqual(oldTaints, newTaints) ||
			!reflect.DeepEqual(oldNode.Spec.Taints, newNode.Spec.Taints) {
			invalidPredicates.Insert(predicates.PodToleratesNodeTaintsPred)
		}

		if !reflect.DeepEqual(oldNode.Status.Conditions, newNode.Status.Conditions) {
			oldConditions := make(map[v1.NodeConditionType]v1.ConditionStatus)
			newConditions := make(map[v1.NodeConditionType]v1.ConditionStatus)
			for _, cond := range oldNode.Status.Conditions {
				oldConditions[cond.Type] = cond.Status
			}
			for _, cond := range newNode.Status.Conditions {
				newConditions[cond.Type] = cond.Status
			}
			if oldConditions[v1.NodeMemoryPressure] != newConditions[v1.NodeMemoryPressure] {
				invalidPredicates.Insert(predicates.CheckNodeMemoryPressurePred)
			}
			if oldConditions[v1.NodeDiskPressure] != newConditions[v1.NodeDiskPressure] {
				invalidPredicates.Insert(predicates.CheckNodeDiskPressurePred)
			}
			if oldConditions[v1.NodePIDPressure] != newConditions[v1.NodePIDPressure] {
				invalidPredicates.Insert(predicates.CheckNodePIDPressurePred)
			}
			if oldConditions[v1.NodeReady] != newConditions[v1.NodeReady] ||
				oldConditions[v1.NodeOutOfDisk] != newConditions[v1.NodeOutOfDisk] ||
				oldConditions[v1.NodeNetworkUnavailable] != newConditions[v1.NodeNetworkUnavailable] {
				invalidPredicates.Insert(predicates.CheckNodeConditionPred)
			}
		}
		if newNode.Spec.Unschedulable != oldNode.Spec.Unschedulable {
			invalidPredicates.Insert(predicates.CheckNodeConditionPred)
		}
		c.equivalencePodCache.InvalidatePredicatesOnNode(newNode.GetName(), invalidPredicates)
	}
}

func (c *configFactory) deleteNodeFromCache(obj interface{}) {
	var node *v1.Node
	switch t := obj.(type) {
	case *v1.Node:
		node = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		node, ok = t.Obj.(*v1.Node)
		if !ok {
			glog.Errorf("cannot convert to *v1.Node: %v", t.Obj)
			return
		}
	default:
		glog.Errorf("cannot convert to *v1.Node: %v", t)
		return
	}
	// NOTE: Because the scheduler uses snapshots of schedulerCache and the live
	// version of equivalencePodCache, updates must be written to schedulerCache
	// before invalidating equivalencePodCache.
	if err := c.schedulerCache.RemoveNode(node); err != nil {
		glog.Errorf("scheduler cache RemoveNode failed: %v", err)
	}
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache.InvalidateAllPredicatesOnNode(node.GetName())
	}
}

func (c *configFactory) addPDBToCache(obj interface{}) {
	pdb, ok := obj.(*v1beta1.PodDisruptionBudget)
	if !ok {
		glog.Errorf("cannot convert to *v1beta1.PodDisruptionBudget: %v", obj)
		return
	}

	if err := c.schedulerCache.AddPDB(pdb); err != nil {
		glog.Errorf("scheduler cache AddPDB failed: %v", err)
	}
}

func (c *configFactory) updatePDBInCache(oldObj, newObj interface{}) {
	oldPDB, ok := oldObj.(*v1beta1.PodDisruptionBudget)
	if !ok {
		glog.Errorf("cannot convert oldObj to *v1beta1.PodDisruptionBudget: %v", oldObj)
		return
	}
	newPDB, ok := newObj.(*v1beta1.PodDisruptionBudget)
	if !ok {
		glog.Errorf("cannot convert newObj to *v1beta1.PodDisruptionBudget: %v", newObj)
		return
	}

	if err := c.schedulerCache.UpdatePDB(oldPDB, newPDB); err != nil {
		glog.Errorf("scheduler cache UpdatePDB failed: %v", err)
	}
}

func (c *configFactory) deletePDBFromCache(obj interface{}) {
	var pdb *v1beta1.PodDisruptionBudget
	switch t := obj.(type) {
	case *v1beta1.PodDisruptionBudget:
		pdb = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pdb, ok = t.Obj.(*v1beta1.PodDisruptionBudget)
		if !ok {
			glog.Errorf("cannot convert to *v1beta1.PodDisruptionBudget: %v", t.Obj)
			return
		}
	default:
		glog.Errorf("cannot convert to *v1beta1.PodDisruptionBudget: %v", t)
		return
	}
	if err := c.schedulerCache.RemovePDB(pdb); err != nil {
		glog.Errorf("scheduler cache RemovePDB failed: %v", err)
	}
}

// Create creates a scheduler with the default algorithm provider.
func (c *configFactory) Create() (*scheduler.Config, error) {
	return c.CreateFromProvider(DefaultProvider)
}

// Creates a scheduler from the name of a registered algorithm provider.
func (c *configFactory) CreateFromProvider(providerName string) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler from algorithm provider '%v'", providerName)
	provider, err := GetAlgorithmProvider(providerName)
	if err != nil {
		return nil, err
	}

	return c.CreateFromKeys(provider.FitPredicateKeys, provider.PriorityFunctionKeys, []algorithm.SchedulerExtender{})
}

// Creates a scheduler from the configuration file
func (c *configFactory) CreateFromConfig(policy schedulerapi.Policy) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler from configuration: %v", policy)

	// validate the policy configuration
	if err := validation.ValidatePolicy(policy); err != nil {
		return nil, err
	}

	predicateKeys := sets.NewString()
	if policy.Predicates == nil {
		glog.V(2).Infof("Using predicates from algorithm provider '%v'", DefaultProvider)
		provider, err := GetAlgorithmProvider(DefaultProvider)
		if err != nil {
			return nil, err
		}
		predicateKeys = provider.FitPredicateKeys
	} else {
		for _, predicate := range policy.Predicates {
			glog.V(2).Infof("Registering predicate: %s", predicate.Name)
			predicateKeys.Insert(RegisterCustomFitPredicate(predicate))
		}
	}

	priorityKeys := sets.NewString()
	if policy.Priorities == nil {
		glog.V(2).Infof("Using priorities from algorithm provider '%v'", DefaultProvider)
		provider, err := GetAlgorithmProvider(DefaultProvider)
		if err != nil {
			return nil, err
		}
		priorityKeys = provider.PriorityFunctionKeys
	} else {
		for _, priority := range policy.Priorities {
			glog.V(2).Infof("Registering priority: %s", priority.Name)
			priorityKeys.Insert(RegisterCustomPriorityFunction(priority))
		}
	}

	var extenders []algorithm.SchedulerExtender
	if len(policy.ExtenderConfigs) != 0 {
		ignoredExtendedResources := sets.NewString()
		for ii := range policy.ExtenderConfigs {
			glog.V(2).Infof("Creating extender with config %+v", policy.ExtenderConfigs[ii])
			extender, err := core.NewHTTPExtender(&policy.ExtenderConfigs[ii])
			if err != nil {
				return nil, err
			}
			extenders = append(extenders, extender)
			for _, r := range policy.ExtenderConfigs[ii].ManagedResources {
				if r.IgnoredByScheduler {
					ignoredExtendedResources.Insert(string(r.Name))
				}
			}
		}
		predicates.RegisterPredicateMetadataProducerWithExtendedResourceOptions(ignoredExtendedResources)
	}
	// Providing HardPodAffinitySymmetricWeight in the policy config is the new and preferred way of providing the value.
	// Give it higher precedence than scheduler CLI configuration when it is provided.
	if policy.HardPodAffinitySymmetricWeight != 0 {
		c.hardPodAffinitySymmetricWeight = policy.HardPodAffinitySymmetricWeight
	}
	// When AlwaysCheckAllPredicates is set to true, scheduler checks all the configured
	// predicates even after one or more of them fails.
	if policy.AlwaysCheckAllPredicates {
		c.alwaysCheckAllPredicates = policy.AlwaysCheckAllPredicates
	}

	return c.CreateFromKeys(predicateKeys, priorityKeys, extenders)
}

// getBinderFunc returns an func which returns an extender that supports bind or a default binder based on the given pod.
func (c *configFactory) getBinderFunc(extenders []algorithm.SchedulerExtender) func(pod *v1.Pod) scheduler.Binder {
	var extenderBinder algorithm.SchedulerExtender
	for i := range extenders {
		if extenders[i].IsBinder() {
			extenderBinder = extenders[i]
			break
		}
	}
	defaultBinder := &binder{c.client}
	return func(pod *v1.Pod) scheduler.Binder {
		if extenderBinder != nil && extenderBinder.IsInterested(pod) {
			return extenderBinder
		}
		return defaultBinder
	}
}

// Creates a scheduler from a set of registered fit predicate keys and priority keys.
func (c *configFactory) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*scheduler.Config, error) {
	glog.V(2).Infof("Creating scheduler with fit predicates '%v' and priority functions '%v'", predicateKeys, priorityKeys)

	if c.GetHardPodAffinitySymmetricWeight() < 1 || c.GetHardPodAffinitySymmetricWeight() > 100 {
		return nil, fmt.Errorf("invalid hardPodAffinitySymmetricWeight: %d, must be in the range 1-100", c.GetHardPodAffinitySymmetricWeight())
	}

	predicateFuncs, err := c.GetPredicates(predicateKeys)
	if err != nil {
		return nil, err
	}

	priorityConfigs, err := c.GetPriorityFunctionConfigs(priorityKeys)
	if err != nil {
		return nil, err
	}

	priorityMetaProducer, err := c.GetPriorityMetadataProducer()
	if err != nil {
		return nil, err
	}

	predicateMetaProducer, err := c.GetPredicateMetadataProducer()
	if err != nil {
		return nil, err
	}

	// Init equivalence class cache
	if c.enableEquivalenceClassCache {
		c.equivalencePodCache = equivalence.NewCache()
		glog.Info("Created equivalence class cache")
	}
	glog.Infof("[tanle] LogComplTimeFeature %s", LogComplTimeFeature)
	algo := core.NewGenericScheduler(
		c.schedulerCache,
		c.equivalencePodCache,
		c.podQueue,
		predicateFuncs,
		predicateMetaProducer,
		priorityConfigs,
		priorityMetaProducer,
		extenders,
		c.volumeBinder,
		c.pVCLister,
		c.alwaysCheckAllPredicates,
		c.disablePreemption,
		c.percentageOfNodesToScore,
		c.client)

	podBackoff := util.CreateDefaultPodBackoff()
	return &scheduler.Config{
		SchedulerCache: c.schedulerCache,
		Ecache:         c.equivalencePodCache,
		// The scheduler only needs to consider schedulable nodes.
		NodeLister:          &nodeLister{c.nodeLister},
		Algorithm:           algo,
		GetBinder:           c.getBinderFunc(extenders),
		PodConditionUpdater: &podConditionUpdater{c.client},
		PodPreemptor:        &podPreemptor{c.client},
		WaitForCacheSync: func() bool {
			return cache.WaitForCacheSync(c.StopEverything, c.scheduledPodsHasSynced)
		},
		NextPod: func() *v1.Pod {
			return c.getNextPod()
		},
		Error: c.MakeDefaultErrorFunc(podBackoff, c.podQueue),
		DeletePodFromQueueingSchedule: c.DeletePod(c.podQueue), // tanle added this function.
		StopEverything:                c.StopEverything,
		VolumeBinder:                  c.volumeBinder,
	}, nil
}

type nodeLister struct {
	corelisters.NodeLister
}

func (n *nodeLister) List() ([]*v1.Node, error) {
	return n.NodeLister.List(labels.Everything())
}

func (c *configFactory) GetPriorityFunctionConfigs(priorityKeys sets.String) ([]algorithm.PriorityConfig, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getPriorityFunctionConfigs(priorityKeys, *pluginArgs)
}

func (c *configFactory) GetPriorityMetadataProducer() (algorithm.PriorityMetadataProducer, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getPriorityMetadataProducer(*pluginArgs)
}

func (c *configFactory) GetPredicateMetadataProducer() (algorithm.PredicateMetadataProducer, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}
	return getPredicateMetadataProducer(*pluginArgs)
}

func (c *configFactory) GetPredicates(predicateKeys sets.String) (map[string]algorithm.FitPredicate, error) {
	pluginArgs, err := c.getPluginArgs()
	if err != nil {
		return nil, err
	}

	return getFitPredicateFunctions(predicateKeys, *pluginArgs)
}

func (c *configFactory) getPluginArgs() (*PluginFactoryArgs, error) {
	return &PluginFactoryArgs{
		PodLister:                      c.podLister,
		ServiceLister:                  c.serviceLister,
		ControllerLister:               c.controllerLister,
		ReplicaSetLister:               c.replicaSetLister,
		StatefulSetLister:              c.statefulSetLister,
		NodeLister:                     &nodeLister{c.nodeLister},
		NodeInfo:                       &predicates.CachedNodeInfo{NodeLister: c.nodeLister},
		PVInfo:                         &predicates.CachedPersistentVolumeInfo{PersistentVolumeLister: c.pVLister},
		PVCInfo:                        &predicates.CachedPersistentVolumeClaimInfo{PersistentVolumeClaimLister: c.pVCLister},
		StorageClassInfo:               &predicates.CachedStorageClassInfo{StorageClassLister: c.storageClassLister},
		VolumeBinder:                   c.volumeBinder,
		HardPodAffinitySymmetricWeight: c.hardPodAffinitySymmetricWeight,
	}, nil
}

// func (c *configFactory) getNextPod() *v1.Pod {
// 	pod, err := c.podQueue.Pop()
// 	if err == nil {
// 		glog.V(4).Infof("About to try and schedule pod %v/%v", pod.Namespace, pod.Name)
// 		return pod
// 	}
// 	glog.Errorf("Error while retrieving next pod from scheduling queue: %v", err)
// 	return nil
// }

func (c *configFactory) getNextPod() *v1.Pod {
	if schedulercache.ENABLE_ONLINE_SCHEDULER {
		// glog.Infof("[meng] getNextPod()")
		if pod := c.podQueue.PickNextPod(c.GetClient()); pod != nil {
			return pod
		} else {
			return nil
		}
	} else {
		pod, err := c.podQueue.Pop()
		if err == nil {
			glog.V(4).Infof("About to try and schedule pod %v/%v", pod.Namespace, pod.Name)
			return pod
		}
		glog.Errorf("Error while retrieving next pod from scheduling queue: %v", err)
	}
	return nil
}

// unassignedNonTerminatedPod selects pods that are unassigned and non-terminal.
func unassignedNonTerminatedPod(pod *v1.Pod) bool {
	if len(pod.Spec.NodeName) != 0 {
		return false
	}
	if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
		return false
	}
	return true
}

// assignedNonTerminatedPod selects pods that are assigned and non-terminal (scheduled and running).
func assignedNonTerminatedPod(pod *v1.Pod) bool {
	if len(pod.Spec.NodeName) == 0 {
		return false
	}
	if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
		return false
	}
	return true
}

// responsibleForPod returns true if the pod has asked to be scheduled by the given scheduler.
func responsibleForPod(pod *v1.Pod, schedulerName string) bool {
	return schedulerName == pod.Spec.SchedulerName
}

// assignedPodLister filters the pods returned from a PodLister to
// only include those that have a node name set.
type assignedPodLister struct {
	corelisters.PodLister
}

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodLister) List(selector labels.Selector) ([]*v1.Pod, error) {
	list, err := l.PodLister.List(selector)
	if err != nil {
		return nil, err
	}
	filtered := make([]*v1.Pod, 0, len(list))
	for _, pod := range list {
		if len(pod.Spec.NodeName) > 0 {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return assignedPodNamespaceLister{l.PodLister.Pods(namespace)}
}

// assignedPodNamespaceLister filters the pods returned from a PodNamespaceLister to
// only include those that have a node name set.
type assignedPodNamespaceLister struct {
	corelisters.PodNamespaceLister
}

// List lists all Pods in the indexer for a given namespace.
func (l assignedPodNamespaceLister) List(selector labels.Selector) (ret []*v1.Pod, err error) {
	list, err := l.PodNamespaceLister.List(selector)
	if err != nil {
		return nil, err
	}
	filtered := make([]*v1.Pod, 0, len(list))
	for _, pod := range list {
		if len(pod.Spec.NodeName) > 0 {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

// Get retrieves the Pod from the indexer for a given namespace and name.
func (l assignedPodNamespaceLister) Get(name string) (*v1.Pod, error) {
	pod, err := l.PodNamespaceLister.Get(name)
	if err != nil {
		return nil, err
	}
	if len(pod.Spec.NodeName) > 0 {
		return pod, nil
	}
	return nil, errors.NewNotFound(schema.GroupResource{Resource: string(v1.ResourcePods)}, name)
}

type podInformer struct {
	informer cache.SharedIndexInformer
}

func (i *podInformer) Informer() cache.SharedIndexInformer {
	return i.informer
}

func (i *podInformer) Lister() corelisters.PodLister {
	return corelisters.NewPodLister(i.informer.GetIndexer())
}

// NewPodInformer creates a shared index informer that returns only non-terminal pods.
func NewPodInformer(client clientset.Interface, resyncPeriod time.Duration) coreinformers.PodInformer {
	selector := fields.ParseSelectorOrDie(
		"status.phase!=" + string(v1.PodSucceeded) +
			",status.phase!=" + string(v1.PodFailed))
	lw := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), string(v1.ResourcePods), metav1.NamespaceAll, selector)
	return &podInformer{
		informer: cache.NewSharedIndexInformer(lw, &v1.Pod{}, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}),
	}
}

func (c *configFactory) MakeDefaultErrorFunc(backoff *util.PodBackoff, podQueue core.SchedulingQueue) func(pod *v1.Pod, err error) {
	return func(pod *v1.Pod, err error) {
		if err == core.ErrNoNodesAvailable {
			glog.V(4).Infof("Unable to schedule %v/%v: no nodes are registered to the cluster; waiting", pod.Namespace, pod.Name)
		} else {
			if _, ok := err.(*core.FitError); ok {
				glog.V(4).Infof("Unable to schedule %v/%v: no fit: %v; waiting", pod.Namespace, pod.Name, err)
			} else if errors.IsNotFound(err) {
				if errStatus, ok := err.(errors.APIStatus); ok && errStatus.Status().Details.Kind == "node" {
					nodeName := errStatus.Status().Details.Name
					// when node is not found, We do not remove the node right away. Trying again to get
					// the node and if the node is still not found, then remove it from the scheduler cache.
					_, err := c.client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
					if err != nil && errors.IsNotFound(err) {
						node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
						// NOTE: Because the scheduler uses snapshots of schedulerCache and the live
						// version of equivalencePodCache, updates must be written to schedulerCache
						// before invalidating equivalencePodCache.
						c.schedulerCache.RemoveNode(&node)
						// invalidate cached predicate for the node
						if c.enableEquivalenceClassCache {
							c.equivalencePodCache.InvalidateAllPredicatesOnNode(nodeName)
						}
					}
				}
			} else {
				glog.Errorf("Error scheduling %v/%v: %v; retrying", pod.Namespace, pod.Name, err)
			}
		}

		backoff.Gc()
		// Retry asynchronously.
		// Note that this is extremely rudimentary and we need a more real error handling path.
		go func() {
			defer runtime.HandleCrash()
			podID := types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			}
			origPod := pod

			// When pod priority is enabled, we would like to place an unschedulable
			// pod in the unschedulable queue. This ensures that if the pod is nominated
			// to run on a node, scheduler takes the pod into account when running
			// predicates for the node.
			if !util.PodPriorityEnabled() {
				entry := backoff.GetEntry(podID)
				if !entry.TryWait(backoff.MaxDuration()) {
					glog.Warningf("Request for pod %v already in flight, abandoning", podID)
					return
				}
			}
			// Get the pod again; it may have changed/been scheduled already.
			getBackoff := initialGetBackoff
			for {
				pod, err := c.client.CoreV1().Pods(podID.Namespace).Get(podID.Name, metav1.GetOptions{})
				if err == nil {
					if len(pod.Spec.NodeName) == 0 {
						podQueue.AddUnschedulableIfNotPresent(pod)
					} else {
						if c.volumeBinder != nil {
							// Volume binder only wants to keep unassigned pods
							c.volumeBinder.DeletePodBindings(pod)
						}
					}
					break
				}
				if errors.IsNotFound(err) {
					glog.Warningf("A pod %v no longer exists", podID)

					if c.volumeBinder != nil {
						// Volume binder only wants to keep unassigned pods
						c.volumeBinder.DeletePodBindings(origPod)
					}
					return
				}
				glog.Errorf("Error getting pod %v for retry: %v; retrying...", podID, err)
				if getBackoff = getBackoff * 2; getBackoff > maximalGetBackoff {
					getBackoff = maximalGetBackoff
				}
				time.Sleep(getBackoff)
			}
		}()
	}
}

// nodeEnumerator allows a cache.Poller to enumerate items in an v1.NodeList
type nodeEnumerator struct {
	*v1.NodeList
}

// Len returns the number of items in the node list.
func (ne *nodeEnumerator) Len() int {
	if ne.NodeList == nil {
		return 0
	}
	return len(ne.Items)
}

// Get returns the item (and ID) with the particular index.
func (ne *nodeEnumerator) Get(index int) interface{} {
	return &ne.Items[index]
}

type binder struct {
	Client clientset.Interface
}

// Bind just does a POST binding RPC.
func (b *binder) Bind(binding *v1.Binding) error {
	glog.V(3).Infof("Attempting to bind %v to %v", binding.Name, binding.Target.Name)
	return b.Client.CoreV1().Pods(binding.Namespace).Bind(binding)
}

type podConditionUpdater struct {
	Client clientset.Interface
}

func (p *podConditionUpdater) Update(pod *v1.Pod, condition *v1.PodCondition) error {
	glog.V(3).Infof("Updating pod condition for %s/%s to (%s==%s)", pod.Namespace, pod.Name, condition.Type, condition.Status)
	if podutil.UpdatePodCondition(&pod.Status, condition) {
		_, err := p.Client.CoreV1().Pods(pod.Namespace).UpdateStatus(pod)
		return err
	}
	return nil
}

type podPreemptor struct {
	Client clientset.Interface
}

func (p *podPreemptor) GetUpdatedPod(pod *v1.Pod) (*v1.Pod, error) {
	return p.Client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
}

func (p *podPreemptor) DeletePod(pod *v1.Pod) error {
	return p.Client.CoreV1().Pods(pod.Namespace).Delete(pod.Name, &metav1.DeleteOptions{})
}

func (p *podPreemptor) SetNominatedNodeName(pod *v1.Pod, nominatedNodeName string) error {
	podCopy := pod.DeepCopy()
	podCopy.Status.NominatedNodeName = nominatedNodeName
	_, err := p.Client.CoreV1().Pods(pod.Namespace).UpdateStatus(podCopy)
	return err
}

func (p *podPreemptor) RemoveNominatedNodeName(pod *v1.Pod) error {
	if len(pod.Status.NominatedNodeName) == 0 {
		return nil
	}
	return p.SetNominatedNodeName(pod, "")
}
