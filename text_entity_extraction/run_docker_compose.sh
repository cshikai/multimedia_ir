# cd elastdocker
# make setup
# cd ..
sysctl -w vm.max_map_count=262144
docker network create -d bridge multimodal
docker-compose -f docker-compose.elastic.yml up -d
docker-compose -f docker-compose.jerex.yml up -d
docker-compose -f docker-compose.blink.yml up -d
