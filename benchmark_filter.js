const tasks = Array.from({ length: 10000 }, (_, i) => ({
    id: i,
    status: i % 2 === 0 ? 'done' : 'todo'
}));

console.time('filter');
for (let i = 0; i < 1000; i++) {
    const activeCount = tasks.filter(t => t.status !== 'done').length;
}
console.timeEnd('filter');

console.time('reduce');
for (let i = 0; i < 1000; i++) {
    const activeCount = tasks.reduce((count, t) => t.status !== 'done' ? count + 1 : count, 0);
}
console.timeEnd('reduce');

console.time('loop');
for (let i = 0; i < 1000; i++) {
    let count = 0;
    for (let j = 0; j < tasks.length; j++) {
        if (tasks[j].status !== 'done') count++;
    }
}
console.timeEnd('loop');
