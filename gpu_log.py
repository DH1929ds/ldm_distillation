import threading
import GPUtil
import time

class GPUMonitor:
    def __init__(self, monitoring_interval=10):
        self.monitoring_interval = monitoring_interval
        self.monitoring_thread = None
        self._stop_event = threading.Event()

    def start(self, text):
        """
        GPU 모니터링을 시작하는 함수입니다.
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self._stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor)
            self.monitoring_thread.start()
            print(f"\n{text}!!!!")
            print(f"{text}!!!!") 
            print(f"{text}!!!!") 
        else:
            print("GPU 모니터링이 이미 실행 중입니다.")

    def stop(self, text):
        """
        GPU 모니터링을 중단하는 함수입니다.
        """
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            self._monitor_end()
            self._stop_event.set()
            self.monitoring_thread.join()
            print(f"{text}!!!!")
            print(f"{text}!!!!") 
            print(f"{text}!!!!\n")  
        else:
            print("GPU 모니터링이 실행 중이 아닙니다.")

    def _monitor(self):
        """
        GPU 상태를 모니터링하는 내부 함수입니다.
        """
        try:
            while not self._stop_event.is_set():
                gpus = GPUtil.getGPUs()
                print("\n########################################################################################")
                for gpu in gpus:
                    print(f"GPU ID: {gpu.id}, Utilization: {gpu.load * 100:.2f}%, memory_used: {gpu.memoryUsed}MB, memory Utilization: {gpu.memoryUtil * 100:.2f}%")
                print("########################################################################################\n")
                time.sleep(self.monitoring_interval)
        except KeyboardInterrupt:
            print("GPU 모니터링을 중단합니다.")

    def _monitor_end(self): 
        """
        GPU 상태를 모니터링하는 내부 함수입니다.
        """
       
        gpus = GPUtil.getGPUs()
        print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}, Utilization: {gpu.load * 100:.2f}%, memory_used: {gpu.memoryUsed}MB, memory Utilization: {gpu.memoryUtil * 100:.2f}%")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")