# Tutorial 11: Smart Pointers & Interior Mutability

## Box<T>: Heap Allocation

Box<T> is the simplest smart pointer, providing heap allocation and ownership.

```rust
// src/main.rs

// When to use Box<T>:
// 1. Large data that you want on the heap
// 2. Recursive types (unknown size at compile time)
// 3. Trait objects

use std::mem;

fn main() {
    // Basic Box usage
    let b = Box::new(5);
    println!("b = {}", b);
    
    // Large struct on heap
    #[derive(Debug)]
    struct LargeData {
        data: [u8; 1000],
    }
    
    let stack_data = LargeData { data: [0; 1000] };
    let heap_data = Box::new(LargeData { data: [1; 1000] });
    
    println!("Stack size: {}", mem::size_of_val(&stack_data));
    println!("Box size: {}", mem::size_of_val(&heap_data));
    
    // Recursive type (linked list)
    #[derive(Debug)]
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }
    
    use List::{Cons, Nil};
    
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("List: {:?}", list);
    
    // Box with trait objects
    trait Draw {
        fn draw(&self);
    }
    
    struct Button {
        label: String,
    }
    
    struct TextField {
        placeholder: String,
    }
    
    impl Draw for Button {
        fn draw(&self) {
            println!("Drawing button: {}", self.label);
        }
    }
    
    impl Draw for TextField {
        fn draw(&self) {
            println!("Drawing text field: {}", self.placeholder);
        }
    }
    
    let components: Vec<Box<dyn Draw>> = vec![
        Box::new(Button { label: "OK".to_string() }),
        Box::new(TextField { placeholder: "Enter name".to_string() }),
    ];
    
    for component in components {
        component.draw();
    }
}
```

## Rc<T>: Reference Counting

Rc<T> enables multiple ownership through reference counting (single-threaded only).

```rust
// src/main.rs
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<Node>>,
}

fn main() {
    // Basic Rc usage
    let a = Rc::new(String::from("hello"));
    println!("Reference count: {}", Rc::strong_count(&a));
    
    let b = Rc::clone(&a); // Doesn't deep clone, just increments count
    println!("Reference count after clone: {}", Rc::strong_count(&a));
    
    {
        let c = Rc::clone(&a);
        println!("Reference count with c: {}", Rc::strong_count(&a));
    } // c goes out of scope, count decrements
    
    println!("Reference count after c dropped: {}", Rc::strong_count(&a));
    
    // Graph/Tree structure with shared nodes
    let leaf = Rc::new(Node {
        value: 3,
        children: vec![],
    });
    
    let branch = Rc::new(Node {
        value: 5,
        children: vec![Rc::clone(&leaf)],
    });
    
    let root = Node {
        value: 10,
        children: vec![Rc::clone(&branch), Rc::clone(&leaf)],
    };
    
    println!("Root: {:?}", root);
    println!("Leaf is shared, count: {}", Rc::strong_count(&leaf));
    
    // Rc with mutable data (requires RefCell)
    use std::cell::RefCell;
    
    let shared_vec = Rc::new(RefCell::new(vec![1, 2, 3]));
    
    let vec1 = Rc::clone(&shared_vec);
    let vec2 = Rc::clone(&shared_vec);
    
    vec1.borrow_mut().push(4);
    vec2.borrow_mut().push(5);
    
    println!("Shared vec: {:?}", shared_vec.borrow());
}
```

## Arc<T>: Atomic Reference Counting

Arc<T> is the thread-safe version of Rc<T>.

```rust
// src/main.rs
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug)]
struct SharedData {
    counter: Mutex<i32>,
    name: String,
}

fn main() {
    // Basic Arc usage
    let data = Arc::new(SharedData {
        counter: Mutex::new(0),
        name: String::from("shared"),
    });
    
    let mut handles = vec![];
    
    for i in 0..5 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let mut counter = data.counter.lock().unwrap();
            *counter += 1;
            println!("Thread {} incremented counter to {}", i, *counter);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final counter: {}", *data.counter.lock().unwrap());
    
    // Arc with immutable data (no Mutex needed)
    let config = Arc::new(vec!["setting1", "setting2", "setting3"]);
    let mut handles = vec![];
    
    for i in 0..3 {
        let config = Arc::clone(&config);
        let handle = thread::spawn(move || {
            println!("Thread {} reading config: {:?}", i, config);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

## RefCell<T>: Interior Mutability

RefCell<T> enables mutation of data even when there are immutable references.

```rust
// src/main.rs
use std::cell::RefCell;
use std::rc::Rc;

// Interior mutability pattern
#[derive(Debug)]
struct MockMessenger {
    sent_messages: RefCell<Vec<String>>,
}

impl MockMessenger {
    fn new() -> MockMessenger {
        MockMessenger {
            sent_messages: RefCell::new(vec![]),
        }
    }
    
    fn send(&self, message: &str) {
        self.sent_messages.borrow_mut().push(String::from(message));
    }
    
    fn sent_count(&self) -> usize {
        self.sent_messages.borrow().len()
    }
}

// Tree with parent references (circular reference example)
#[derive(Debug)]
struct TreeNode {
    value: i32,
    parent: RefCell<Option<Rc<TreeNode>>>,
    children: RefCell<Vec<Rc<TreeNode>>>,
}

fn main() {
    // Basic RefCell usage
    let x = RefCell::new(5);
    
    {
        let borrowed = x.borrow();
        println!("Borrowed value: {}", *borrowed);
        // let mut_borrowed = x.borrow_mut(); // Would panic! Already borrowed
    }
    
    {
        let mut borrowed_mut = x.borrow_mut();
        *borrowed_mut += 10;
    }
    
    println!("Modified value: {}", x.borrow());
    
    // Mock object pattern
    let messenger = MockMessenger::new();
    messenger.send("Hello");
    messenger.send("World");
    
    println!("Messages sent: {}", messenger.sent_count());
    println!("Messages: {:?}", messenger.sent_messages.borrow());
    
    // Rc<RefCell<T>> pattern for shared mutable state
    let shared_list = Rc::new(RefCell::new(vec![1, 2, 3]));
    
    let list1 = Rc::clone(&shared_list);
    let list2 = Rc::clone(&shared_list);
    
    list1.borrow_mut().push(4);
    println!("List after first push: {:?}", shared_list.borrow());
    
    list2.borrow_mut().push(5);
    println!("List after second push: {:?}", shared_list.borrow());
}
```

## Weak<T>: Breaking Reference Cycles

Weak<T> prevents reference cycles that would cause memory leaks.

```rust
// src/main.rs
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<Self> {
        Rc::new(Node {
            value,
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(vec![]),
        })
    }
    
    fn add_child(parent: &Rc<Node>, child: Rc<Node>) {
        child.parent.replace(Rc::downgrade(parent));
        parent.children.borrow_mut().push(child);
    }
}

// Observer pattern with weak references
struct Subject {
    observers: RefCell<Vec<Weak<dyn Observer>>>,
}

trait Observer {
    fn update(&self, value: i32);
}

struct ConcreteObserver {
    id: i32,
}

impl Observer for ConcreteObserver {
    fn update(&self, value: i32) {
        println!("Observer {} received update: {}", self.id, value);
    }
}

impl Subject {
    fn new() -> Self {
        Subject {
            observers: RefCell::new(vec![]),
        }
    }
    
    fn attach(&self, observer: Weak<dyn Observer>) {
        self.observers.borrow_mut().push(observer);
    }
    
    fn notify(&self, value: i32) {
        let mut observers = self.observers.borrow_mut();
        
        // Remove dead weak references
        observers.retain(|observer| observer.strong_count() > 0);
        
        for observer in observers.iter() {
            if let Some(observer) = observer.upgrade() {
                observer.update(value);
            }
        }
    }
}

fn main() {
    // Tree structure with parent references
    let root = Node::new(5);
    let branch = Node::new(3);
    let leaf = Node::new(1);
    
    Node::add_child(&root, Rc::clone(&branch));
    Node::add_child(&branch, leaf);
    
    println!("Root strong count: {}", Rc::strong_count(&root));
    println!("Branch strong count: {}", Rc::strong_count(&branch));
    
    // Access parent through weak reference
    if let Some(parent) = branch.parent.borrow().upgrade() {
        println!("Branch's parent value: {}", parent.value);
    }
    
    // Observer pattern example
    let subject = Rc::new(Subject::new());
    
    let observer1 = Rc::new(ConcreteObserver { id: 1 });
    let observer2 = Rc::new(ConcreteObserver { id: 2 });
    
    subject.attach(Rc::downgrade(&observer1) as Weak<dyn Observer>);
    subject.attach(Rc::downgrade(&observer2) as Weak<dyn Observer>);
    
    subject.notify(42);
    
    // Drop observer1
    drop(observer1);
    
    println!("\nAfter dropping observer1:");
    subject.notify(100);
}
```

## Cell<T>: Copy Types Interior Mutability

Cell<T> provides interior mutability for Copy types without runtime borrowing checks.

```rust
// src/main.rs
use std::cell::Cell;

struct Counter {
    value: Cell<i32>,
}

impl Counter {
    fn new() -> Self {
        Counter {
            value: Cell::new(0),
        }
    }
    
    fn increment(&self) {
        self.value.set(self.value.get() + 1);
    }
    
    fn get(&self) -> i32 {
        self.value.get()
    }
}

// Game state example
struct GameState {
    score: Cell<u32>,
    lives: Cell<u32>,
    level: Cell<u32>,
}

impl GameState {
    fn new() -> Self {
        GameState {
            score: Cell::new(0),
            lives: Cell::new(3),
            level: Cell::new(1),
        }
    }
    
    fn add_score(&self, points: u32) {
        self.score.set(self.score.get() + points);
    }
    
    fn lose_life(&self) {
        let current = self.lives.get();
        if current > 0 {
            self.lives.set(current - 1);
        }
    }
    
    fn next_level(&self) {
        self.level.set(self.level.get() + 1);
        self.lives.set(3); // Reset lives
    }
}

fn main() {
    // Basic Cell usage
    let counter = Counter::new();
    counter.increment();
    counter.increment();
    println!("Counter: {}", counter.get());
    
    // Cell in struct
    let game = GameState::new();
    game.add_score(100);
    game.lose_life();
    game.add_score(50);
    
    println!("Score: {}, Lives: {}, Level: {}", 
        game.score.get(), 
        game.lives.get(), 
        game.level.get()
    );
    
    game.next_level();
    println!("After next level - Score: {}, Lives: {}, Level: {}", 
        game.score.get(), 
        game.lives.get(), 
        game.level.get()
    );
}
```

## Real-World Example: Cache with Multiple Access Patterns

```rust
// src/main.rs
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

// Single-threaded cache with Rc and RefCell
struct LocalCache<K, V> {
    data: Rc<RefCell<HashMap<K, V>>>,
    stats: Rc<RefCell<CacheStats>>,
}

// Thread-safe cache with Arc and RwLock
struct SharedCache<K, V> {
    data: Arc<RwLock<HashMap<K, V>>>,
    stats: Arc<Mutex<CacheStats>>,
}

#[derive(Debug, Default)]
struct CacheStats {
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> LocalCache<K, V> {
    fn new() -> Self {
        LocalCache {
            data: Rc::new(RefCell::new(HashMap::new())),
            stats: Rc::new(RefCell::new(CacheStats::default())),
        }
    }
    
    fn get(&self, key: &K) -> Option<V> {
        let cache = self.data.borrow();
        let mut stats = self.stats.borrow_mut();
        
        match cache.get(key) {
            Some(value) => {
                stats.hits += 1;
                Some(value.clone())
            }
            None => {
                stats.misses += 1;
                None
            }
        }
    }
    
    fn insert(&self, key: K, value: V) {
        self.data.borrow_mut().insert(key, value);
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.borrow().clone()
    }
}

impl<K: Eq + std::hash::Hash + Clone + Send + Sync, V: Clone + Send + Sync> 
    SharedCache<K, V> 
{
    fn new() -> Self {
        SharedCache {
            data: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }
    
    fn get(&self, key: &K) -> Option<V> {
        let cache = self.data.read().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        match cache.get(key) {
            Some(value) => {
                stats.hits += 1;
                Some(value.clone())
            }
            None => {
                stats.misses += 1;
                None
            }
        }
    }
    
    fn insert(&self, key: K, value: V) {
        self.data.write().unwrap().insert(key, value);
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }
}

// Advanced: LRU Cache with Weak references for memory pressure handling
struct LruEntry<K, V> {
    key: K,
    value: V,
    prev: Option<Weak<RefCell<LruEntry<K, V>>>>,
    next: Option<Rc<RefCell<LruEntry<K, V>>>>,
}

struct LruCache<K, V> {
    capacity: usize,
    map: RefCell<HashMap<K, Rc<RefCell<LruEntry<K, V>>>>>,
    head: Option<Rc<RefCell<LruEntry<K, V>>>>,
    tail: Option<Weak<RefCell<LruEntry<K, V>>>>,
}

// Event system with weak observers
struct EventEmitter<T> {
    listeners: RefCell<Vec<Weak<dyn Fn(&T)>>>,
}

impl<T> EventEmitter<T> {
    fn new() -> Self {
        EventEmitter {
            listeners: RefCell::new(Vec::new()),
        }
    }
    
    fn subscribe(&self, listener: Weak<dyn Fn(&T)>) {
        self.listeners.borrow_mut().push(listener);
    }
    
    fn emit(&self, event: &T) {
        let mut listeners = self.listeners.borrow_mut();
        listeners.retain(|listener| listener.strong_count() > 0);
        
        for listener in listeners.iter() {
            if let Some(listener) = listener.upgrade() {
                listener(event);
            }
        }
    }
}

fn main() {
    // Local cache example
    println!("=== Local Cache ===");
    let local_cache = LocalCache::new();
    
    local_cache.insert("key1", "value1");
    local_cache.insert("key2", "value2");
    
    println!("Get key1: {:?}", local_cache.get(&"key1"));
    println!("Get key3: {:?}", local_cache.get(&"key3"));
    
    let stats = local_cache.stats();
    println!("Stats: {:?}", stats);
    
    // Shared cache example
    println!("\n=== Shared Cache ===");
    let shared_cache = Arc::new(SharedCache::new());
    let mut handles = vec![];
    
    // Multiple threads accessing cache
    for i in 0..5 {
        let cache = Arc::clone(&shared_cache);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let key = format!("key{}", j);
                let value = format!("value{}-{}", i, j);
                
                if j % 2 == 0 {
                    cache.insert(key.clone(), value);
                } else {
                    cache.get(&key);
                }
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let stats = shared_cache.stats();
    println!("Shared cache stats: {:?}", stats);
    
    // Event emitter example
    println!("\n=== Event System ===");
    let emitter = Rc::new(EventEmitter::new());
    
    let listener1 = Rc::new(|event: &String| {
        println!("Listener 1 received: {}", event);
    });
    
    let listener2 = Rc::new(|event: &String| {
        println!("Listener 2 received: {}", event);
    });
    
    emitter.subscribe(Rc::downgrade(&listener1) as Weak<dyn Fn(&String)>);
    emitter.subscribe(Rc::downgrade(&listener2) as Weak<dyn Fn(&String)>);
    
    emitter.emit(&String::from("First event"));
    
    drop(listener1); // Remove listener1
    
    emitter.emit(&String::from("Second event"));
}
```

## Memory Patterns and Best Practices

```rust
// src/main.rs
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};

// Pattern 1: Shared ownership with mutation
struct SharedState {
    data: Rc<RefCell<Vec<i32>>>,
}

impl SharedState {
    fn new() -> Self {
        SharedState {
            data: Rc::new(RefCell::new(Vec::new())),
        }
    }
    
    fn clone_handle(&self) -> Self {
        SharedState {
            data: Rc::clone(&self.data),
        }
    }
    
    fn add(&self, value: i32) {
        self.data.borrow_mut().push(value);
    }
    
    fn sum(&self) -> i32 {
        self.data.borrow().iter().sum()
    }
}

// Pattern 2: Tree with parent pointers
#[derive(Debug)]
struct TreeNode<T> {
    value: T,
    parent: RefCell<Option<Weak<TreeNode<T>>>>,
    children: RefCell<Vec<Rc<TreeNode<T>>>>,
}

impl<T> TreeNode<T> {
    fn new(value: T) -> Rc<Self> {
        Rc::new(TreeNode {
            value,
            parent: RefCell::new(None),
            children: RefCell::new(Vec::new()),
        })
    }
    
    fn add_child(parent: &Rc<TreeNode<T>>, child: Rc<TreeNode<T>>) {
        child.parent.replace(Some(Rc::downgrade(parent)));
        parent.children.borrow_mut().push(child);
    }
    
    fn ancestors(&self) -> Vec<Rc<TreeNode<T>>> {
        let mut ancestors = Vec::new();
        let mut current = self.parent.borrow().clone();
        
        while let Some(weak) = current {
            if let Some(parent) = weak.upgrade() {
                ancestors.push(Rc::clone(&parent));
                current = parent.parent.borrow().clone();
            } else {
                break;
            }
        }
        
        ancestors
    }
}

// Pattern 3: Thread-safe singleton
struct Singleton {
    data: String,
}

impl Singleton {
    fn instance() -> Arc<Mutex<Singleton>> {
        static mut INSTANCE: Option<Arc<Mutex<Singleton>>> = None;
        static ONCE: std::sync::Once = std::sync::Once::new();
        
        unsafe {
            ONCE.call_once(|| {
                INSTANCE = Some(Arc::new(Mutex::new(Singleton {
                    data: String::from("Singleton instance"),
                })));
            });
            INSTANCE.clone().unwrap()
        }
    }
}

// Pattern 4: Smart pointer choice guide
fn smart_pointer_guide() {
    println!("Smart Pointer Selection Guide:");
    println!("1. Box<T> - Simple heap allocation, single owner");
    println!("2. Rc<T> - Multiple owners, single-threaded");
    println!("3. Arc<T> - Multiple owners, thread-safe");
    println!("4. Cell<T> - Interior mutability for Copy types");
    println!("5. RefCell<T> - Interior mutability with borrowing rules");
    println!("6. Mutex<T> - Thread-safe interior mutability");
    println!("7. RwLock<T> - Thread-safe, multiple readers or one writer");
}

fn main() {
    // Shared state pattern
    let state1 = SharedState::new();
    let state2 = state1.clone_handle();
    
    state1.add(10);
    state2.add(20);
    
    println!("Sum: {}", state1.sum());
    
    // Tree pattern
    let root = TreeNode::new("root");
    let child1 = TreeNode::new("child1");
    let child2 = TreeNode::new("child2");
    let grandchild = TreeNode::new("grandchild");
    
    TreeNode::add_child(&root, Rc::clone(&child1));
    TreeNode::add_child(&root, Rc::clone(&child2));
    TreeNode::add_child(&child1, grandchild.clone());
    
    println!("\nAncestors of grandchild:");
    for ancestor in grandchild.ancestors() {
        println!("  {}", ancestor.value);
    }
    
    // Singleton pattern
    let singleton1 = Singleton::instance();
    let singleton2 = Singleton::instance();
    
    println!("\nSingleton: {}", singleton1.lock().unwrap().data);
    
    smart_pointer_guide();
}
```

## Exercises

1. **Graph Structure**: Implement a directed graph using `Rc` and `Weak` that supports cycle detection and path finding.

2. **Thread Pool with Shared State**: Build a thread pool where workers share a job queue using `Arc<Mutex<VecDeque<Job>>>`.

3. **Observable Pattern**: Create a full observer pattern implementation using `Weak` references to prevent memory leaks.

4. **Caching System**: Implement an LRU cache with both single-threaded (`Rc`/`RefCell`) and multi-threaded (`Arc`/`RwLock`) versions.

5. **Tree Editor**: Build a tree data structure that supports undo/redo operations using smart pointers.

## Key Takeaways

- `Box<T>` provides simple heap allocation with single ownership
- `Rc<T>` enables multiple ownership in single-threaded contexts
- `Arc<T>` is the thread-safe version of `Rc<T>`
- `RefCell<T>` provides runtime borrowing checks for interior mutability
- `Cell<T>` offers interior mutability for `Copy` types without borrowing
- `Weak<T>` breaks reference cycles to prevent memory leaks
- Choose thread-safe types (`Arc`, `Mutex`) only when needed
- Interior mutability allows mutation through immutable references
- Reference counting has runtime overhead
- Always consider ownership patterns when designing data structures

## Next Steps

In Tutorial 12, we'll explore **Macros & Metaprogramming**, learning how to write code that generates code and create powerful abstractions in Rust.