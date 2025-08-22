const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Starting dev server in background...');

// Create log files for debugging
const logFile = path.join(__dirname, 'dev-server.log');
const errorFile = path.join(__dirname, 'dev-server-error.log');
const pidFile = path.join(__dirname, 'dev-server.pid');

// Function to kill processes on port 3000
function killPort3000() {
    return new Promise((resolve) => {
        exec('npx kill-port 3000', (error, stdout, stderr) => {
            if (stdout && stdout.trim()) {
                console.log('✓ Killed existing process on port 3000');
            }
            resolve();
        });
    });
}

// Function to start the server
async function startServer() {
    // First, kill any existing process on port 3000
    await killPort3000();
    
    // Wait a moment for the port to be fully released
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Clear old log files and open file descriptors
    fs.writeFileSync(logFile, '');
    fs.writeFileSync(errorFile, '');

    const logFd = fs.openSync(logFile, 'a');
    const errorFd = fs.openSync(errorFile, 'a');

    const server = spawn('cmd', ['/c', 'npm run dev'], {
        cwd: __dirname,
        detached: true,
        stdio: ['ignore', logFd, errorFd],
        shell: false
    });

    // Store the PID for later cleanup
    fs.writeFileSync(pidFile, server.pid.toString());

    // Close file descriptors after spawn
    fs.closeSync(logFd);
    fs.closeSync(errorFd);

    // Unref so the parent can exit
    server.unref();

    console.log('✓ Dev server started in background on http://localhost:3000');
    console.log(`✓ Process PID: ${server.pid} (saved to dev-server.pid)`);
    console.log(`✓ Logs: dev-server.log, Errors: dev-server-error.log`);
    console.log('Use "node stop-server.js" to stop it');

    // Exit immediately
    process.exit(0);
}

// Run the startup process
startServer().catch(err => {
    console.error('Failed to start server:', err);
    process.exit(1);
});