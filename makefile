lenet: lenet.c main.c lenet_forward.c forward_func.c cnnapi_base.c cnnapi_base_q.c utils.c
	gcc lenet.c main.c lenet_forward.c forward_func.c cnnapi_base.c cnnapi_base_q.c utils.c -lm -o lenet -g

download:
	wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	gzip -d t10k-labels-idx1-ubyte.gz
	gzip -d t10k-images-idx3-ubyte.gz
	gzip -d train-images-idx3-ubyte.gz
	gzip -d train-labels-idx1-ubyte.gz

run:
	./lenet

d:
	gdb lenet

clean:
	rm lenet
