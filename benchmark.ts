const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const mockOnAddTask = async (task: any) => {
    await delay(50); // mock 50ms delay for each task addition
    return true;
};

const results = {
    action_items: Array.from({ length: 20 }, (_, i) => `Task ${i}`)
};

async function runSequential() {
    console.time('Sequential');
    let addedCount = 0;
    for (const item of results.action_items) {
        const success = await mockOnAddTask({
            title: item,
            priority: 'medium',
            dueDate: null,
            status: 'todo',
            sessionId: 1,
            sessionName: 'Test Session'
        });
        if (success) addedCount++;
    }
    console.timeEnd('Sequential');
    return addedCount;
}

async function runConcurrent() {
    console.time('Concurrent');
    const addPromises = results.action_items.map(item =>
        mockOnAddTask({
            title: item,
            priority: 'medium',
            dueDate: null,
            status: 'todo',
            sessionId: 1,
            sessionName: 'Test Session'
        })
    );

    const addResults = await Promise.all(addPromises);
    const addedCount = addResults.filter(success => success).length;
    console.timeEnd('Concurrent');
    return addedCount;
}

async function main() {
    await runSequential();
    await runConcurrent();
}

main();
