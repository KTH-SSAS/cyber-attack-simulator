# infra

This folder contains image and infrastructure management scripts and configuration.

Below is a brief inventory of what each script does.
Most of them follow the same `+`/`-`/`=` convention
for `start`/`stop`/`status`, which can be useful to know,
in case no explicit "usage" information is printed when
a script is called without any command-line arguments.

BUT, before that, the following commands should be available
(unless otherwise noted, use `apt`, `snap`, or `brew` to install them):
- `jq`
- `yq` (`pip install python-yq`)
- `gcloud` (https://cloud.google.com/sdk/docs/install)
- `kubectl`
- `helm`
- `svn` [TEMPORARILY (until Ray publishes their Helm chart)]


## Artifacts

*  [`kas`](kas)
    - builds and pushes Docker images using [`Dockerfile`](Dockerfile)
    - launches commands in a Docker container or on [pre-existing] Kubernetes cluster
    - takes care of port-forwarding, environment variables, etc.

*  [`publish-wheel`](publish-wheel)
    - builds and publishes a Python wheel for the project


## Cloud Infrastructure

on the Google Cloud Platform (GCP).
Other abbreviations used below include:
- Google Compute Engine (GCE)
- Google Kubernetes Engine (GKE)
- Google Artifact Registry (GAR)

*  [`admin-service-account`](admin-service-account)
    - _**requires** `Project IAM Admin` privileges!_
    - manages a service account with the necessary roles to administer GCE, GKE, and GAR
    - configures a `gcloud` profile with said service account and other relevant defaults

*  [`artifact-registry`](artifact-registry)
    - manages two repos in GAR: one for Docker `images` and one for PyPI `wheels`


### Kubernetes (GKE)

*  [`gke-cluster`](gke-cluster)
    - manages the Kubernetes cluster (called `x-ray`) underlying any "Ray on Kubernetes" cluster

*  [`gke-nat`](gke-nat)
    - manages a NAT that can allow pods to connect to the outside world

*  [`gke-allow`](gke-allow)
    - manages the IP addresses that are allowed to connect to the Kubernetes API


#### Ray on Kubernetes

*  [`ray-k8s-cluster`](ray-k8s-cluster)
    - manages Ray clusters [and the Ray operator] theoretically on any Kubernetes cluster  
      (e.g. on top of `x-ray` in GKE or locally in Docker Desktop or minikube clusters)  
    - uses "infra-as-code" definitions, e.g. those provided in the [`yaml/k8s`](yaml/k8s) folder  
      (cf. https://docs.ray.io/en/latest/cluster/kubernetes.html)


### Ray Autoscaler (GCE)

*  [`ray-service-account`](ray-service-account) (needed only when using Docker images)
    - manages Ray service accounts (primarily, for worker nodes) to allow read access to GAR

*  Ray's built-in `ray` command can be used to `start`/`stop` (and `submit` jobs to) the cluster.  
   [Among other things...](https://docs.ray.io/en/latest/package-ref.html?highlight=ray%20command%20reference#the-ray-command-line-api)

   - Ray's autoscaler (like many GCP-enabled applications) uses Google's API client library,
     which in turn needs service account credentials
     ```console
     mehes@mehes-mba:infra$ pip install google-api-python-client
     mehes@mehes-mba:infra$ source admin-service-account
     mehes@mehes-mba:infra$ export GOOGLE_APPLICATION_CREDENTIALS="$KEYFILE"
     ```

   - Sample cluster definitions can be found in
     * the [`yaml/gcp`](yaml/gcp) folder in this repository
     * https://github.com/ray-project/ray/tree/master/python/ray/autoscaler/gcp
