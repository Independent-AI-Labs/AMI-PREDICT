const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Killing all processes on ports 3000-3010...');

const pidFile = path.join(__dirname, 'dev-server.pid');
const logFile = path.join(__dirname, 'dev-server.log');
const errorFile = path.join(__dirname, 'dev-server-error.log');

// Kill all ports from 3000 to 3010 using npx
const ports = Array.from({ length: 11 }, (_, i) => 3000 + i);
const portList = ports.join(' ');

exec(`npx kill-port ${portList}`, (error, stdout, stderr) => {
    if (stdout) {
        console.log(stdout);
    }
    console.log('✓ All processes on ports 3000-3010 killed');
    
    // Clean up files
    [pidFile, logFile, errorFile].forEach(file => {
        if (fs.existsSync(file)) {
            try {
                fs.unlinkSync(file);
            } catch (e) {
                console.log(`Could not delete ${file}`);
            }
        }
    });
    
    console.log('✓ Cleanup complete');
    process.exit(0);
});