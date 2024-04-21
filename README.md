
kubectl run -it busybox --image radial/busyboxplus:curl
curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' https://example.com/api