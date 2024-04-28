
kubectl exec -it PODNAME -- bash
apt install curl

kubectl run -it busybox --image radial/busyboxplus:curl


curl -X POST -H "Content-Type: application/json" -d '{"prompt":"smiling dog in forest"}' http://127.0.0.1:8000/generate --verbose