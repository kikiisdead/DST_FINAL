#ifndef CIRCULAR_BUFFER
#define CiRCULAR_BUFFER

template <typename T>
class CircularBuffer {
private:
    int size;
    int write_ptr;
    T* array;
    
public:
    CircularBuffer(int _size) {
        array = new T[_size]{};
        size = _size;
        write_ptr = 0;
    }

    ~CircularBuffer() {
        delete array;
    }

    void write(T data) {
        array[write_ptr] = data;
        write_ptr = (write_ptr + 1) % size;
    }

    T read(int index) {
        int idx = (write_ptr + index + size) % size;
        return array[idx];
    }

    T& operator[](int index) {
        int idx = (write_ptr + index + size) % size;
        return array[idx];
    }
};

#endif