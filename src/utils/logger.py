import time
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

class Logger:
    """Simple colorful logger"""
    
    @staticmethod
    def info(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.BLUE}ℹ️  [{timestamp}] {msg}{Style.RESET_ALL}", flush=True)
    
    @staticmethod
    def success(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.GREEN}✅ [{timestamp}] {msg}{Style.RESET_ALL}", flush=True)
    
    @staticmethod
    def warning(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.YELLOW}⚠️  [{timestamp}] {msg}{Style.RESET_ALL}", flush=True)
    
    @staticmethod
    def error(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.RED}❌ [{timestamp}] {msg}{Style.RESET_ALL}", flush=True)
    
    @staticmethod
    def progress(msg, percent=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if percent:
            print(f"{Fore.CYAN}📊 [{timestamp}] {msg}: {percent}%{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.CYAN}📊 [{timestamp}] {msg}{Style.RESET_ALL}", flush=True)
    
    @staticmethod
    def section(title):
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n", flush=True)


class ProgressTracker:
    """Track progress at key milestones (25%, 50%, 75%, 100%)"""
    
    def __init__(self, name, file_size_mb=None):
        self.name = name
        self.file_size_mb = file_size_mb
        self.milestones = set()
        self.start = None
    
    def start_timer(self):
        """Start timing"""
        self.start = time.time()
    
    def report(self, percent):
        """Report milestone"""
        milestone = (percent // 25) * 25
        if milestone > 0 and milestone not in self.milestones:
            self.milestones.add(milestone)
            
            elapsed = time.time() - self.start if self.start else 0
            size_str = f" ({self.file_size_mb:.1f} MB)" if self.file_size_mb else ""
            time_str = f" in {elapsed:.1f}s" if milestone == 100 else ""
            
            Logger.progress(f"{self.name}{size_str}", milestone)
            if milestone == 100:
                Logger.success(f"{self.name} completed{time_str}")
