#start server
#gnome-terminal -e command ./simulator --host=127.0.0.1 --port=11200 --viewsize=120x90

gnome-terminal -x bash -c "cd mash-simulator/build/bin && ./simulator --host=127.0.0.1 --port=11200 --viewsize=120x90"

sleep 3

gnome-terminal -x bash -c "./ex/client"
