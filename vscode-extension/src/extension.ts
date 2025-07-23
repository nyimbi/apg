/**
 * APG Language Support Extension for VS Code
 * ==========================================
 * 
 * Main extension file providing APG language support including:
 * - Language Server Protocol integration
 * - Compilation commands
 * - Project management
 * - Syntax validation
 * - Code preview
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec, spawn, ChildProcess } from 'child_process';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

/**
 * Extension activation
 */
export function activate(context: vscode.ExtensionContext) {
    console.log('APG Language Support extension is activating...');
    
    // Create output channel
    outputChannel = vscode.window.createOutputChannel('APG Language');
    context.subscriptions.push(outputChannel);
    
    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(gear) APG";
    statusBarItem.tooltip = "APG Language Support";
    statusBarItem.command = 'apg.showMenu';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
    
    // Register commands
    registerCommands(context);
    
    // Start language server
    startLanguageServer(context);
    
    // Set up file watchers
    setupFileWatchers(context);
    
    outputChannel.appendLine('APG Language Support extension activated');
    console.log('APG Language Support extension is now active!');
}

/**
 * Extension deactivation
 */
export function deactivate(): Thenable<void> | undefined {
    outputChannel.appendLine('APG Language Support extension deactivating...');
    
    if (client) {
        return client.stop();
    }
    
    return undefined;
}

/**
 * Register extension commands
 */
function registerCommands(context: vscode.ExtensionContext) {
    // Compile current file
    const compileCommand = vscode.commands.registerCommand('apg.compile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }
        
        await compileFile(editor.document.uri);
    });
    
    // Compile project
    const compileProjectCommand = vscode.commands.registerCommand('apg.compileProject', async () => {
        const workspaceFolder = getWorkspaceFolder();
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder is open');
            return;
        }
        
        await compileProject(workspaceFolder);
    });
    
    // Run generated application
    const runGeneratedCommand = vscode.commands.registerCommand('apg.runGenerated', async () => {
        const workspaceFolder = getWorkspaceFolder();
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder is open');
            return;
        }
        
        await runGeneratedApp(workspaceFolder);
    });
    
    // Validate syntax
    const validateCommand = vscode.commands.registerCommand('apg.validateSyntax', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }
        
        await validateSyntax(editor.document.uri);
    });
    
    // Show preview
    const previewCommand = vscode.commands.registerCommand('apg.showPreview', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }
        
        await showPreview(editor.document.uri);
    });
    
    // Create new project
    const createProjectCommand = vscode.commands.registerCommand('apg.createProject', async () => {
        await createNewProject();
    });
    
    // Restart language server
    const restartLSCommand = vscode.commands.registerCommand('apg.restartLanguageServer', async () => {
        await restartLanguageServer(context);
    });
    
    // Show menu
    const showMenuCommand = vscode.commands.registerCommand('apg.showMenu', async () => {
        showAPGMenu();
    });
    
    // Register all commands
    context.subscriptions.push(
        compileCommand,
        compileProjectCommand,
        runGeneratedCommand,
        validateCommand,
        previewCommand,
        createProjectCommand,
        restartLSCommand,
        showMenuCommand
    );
}

/**
 * Start the APG Language Server
 */
function startLanguageServer(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('apg.languageServer');
    
    if (!config.get('enabled', true)) {
        outputChannel.appendLine('Language Server is disabled in configuration');
        return;
    }
    
    const host = config.get('host', '127.0.0.1');
    const port = config.get('port', 2087);
    
    // Server options for TCP connection
    const serverOptions: ServerOptions = {
        run: {
            command: 'apg-language-server',
            args: ['--host', host, '--port', port.toString()],
            transport: TransportKind.socket,
            port: port
        },
        debug: {
            command: 'apg-language-server',
            args: ['--host', host, '--port', port.toString(), '--log-level', 'DEBUG'],
            transport: TransportKind.socket,
            port: port
        }
    };
    
    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'apg' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.apg')
        },
        outputChannel: outputChannel,
        revealOutputChannelOn: 4 // Never automatically reveal
    };
    
    // Create and start the language client
    client = new LanguageClient(
        'apg-language-server',
        'APG Language Server',
        serverOptions,
        clientOptions
    );
    
    // Start the client and server
    client.start().then(() => {
        outputChannel.appendLine('APG Language Server started successfully');
        updateStatusBar('connected');
    }).catch((error) => {
        outputChannel.appendLine(`Failed to start APG Language Server: ${error}`);
        updateStatusBar('error');
        vscode.window.showWarningMessage(
            'APG Language Server failed to start. Some features may not be available.',
            'Retry'
        ).then((selection) => {
            if (selection === 'Retry') {
                restartLanguageServer(context);
            }
        });
    });
    
    context.subscriptions.push(client);
}

/**
 * Set up file watchers for APG files
 */
function setupFileWatchers(context: vscode.ExtensionContext) {
    // Watch for changes to APG files
    const apgWatcher = vscode.workspace.createFileSystemWatcher('**/*.apg');
    
    apgWatcher.onDidChange((uri) => {
        outputChannel.appendLine(`APG file changed: ${uri.fsPath}`);
    });
    
    apgWatcher.onDidCreate((uri) => {
        outputChannel.appendLine(`APG file created: ${uri.fsPath}`);
    });
    
    apgWatcher.onDidDelete((uri) => {
        outputChannel.appendLine(`APG file deleted: ${uri.fsPath}`);
    });
    
    context.subscriptions.push(apgWatcher);
    
    // Watch for changes to APG project configuration
    const configWatcher = vscode.workspace.createFileSystemWatcher('**/apg.json');
    
    configWatcher.onDidChange((uri) => {
        outputChannel.appendLine(`APG project configuration changed: ${uri.fsPath}`);
        vscode.window.showInformationMessage(
            'APG project configuration changed. Restart language server?',
            'Restart'
        ).then((selection) => {
            if (selection === 'Restart') {
                restartLanguageServer(context);
            }
        });
    });
    
    context.subscriptions.push(configWatcher);
}

/**
 * Compile a single APG file
 */
async function compileFile(uri: vscode.Uri): Promise<void> {
    outputChannel.show(true);
    outputChannel.appendLine(`Compiling APG file: ${uri.fsPath}`);
    
    updateStatusBar('compiling');
    
    return new Promise((resolve, reject) => {
        const workspaceDir = path.dirname(uri.fsPath);
        const command = `apg build --verbose`;
        
        exec(command, { cwd: workspaceDir }, (error, stdout, stderr) => {
            if (error) {
                outputChannel.appendLine(`Compilation failed: ${error.message}`);
                outputChannel.appendLine(stderr);
                updateStatusBar('error');
                vscode.window.showErrorMessage(`APG compilation failed: ${error.message}`);
                reject(error);
            } else {
                outputChannel.appendLine('Compilation successful!');
                outputChannel.appendLine(stdout);
                updateStatusBar('connected');
                vscode.window.showInformationMessage('APG compilation completed successfully');
                resolve();
            }
        });
    });
}

/**
 * Compile entire APG project
 */
async function compileProject(workspaceFolder: vscode.WorkspaceFolder): Promise<void> {
    outputChannel.show(true);
    outputChannel.appendLine(`Compiling APG project: ${workspaceFolder.uri.fsPath}`);
    
    updateStatusBar('compiling');
    
    return new Promise((resolve, reject) => {
        const command = `apg build --verbose`;
        
        exec(command, { cwd: workspaceFolder.uri.fsPath }, (error, stdout, stderr) => {
            if (error) {
                outputChannel.appendLine(`Project compilation failed: ${error.message}`);
                outputChannel.appendLine(stderr);
                updateStatusBar('error');
                vscode.window.showErrorMessage(`APG project compilation failed: ${error.message}`);
                reject(error);
            } else {
                outputChannel.appendLine('Project compilation successful!');
                outputChannel.appendLine(stdout);
                updateStatusBar('connected');
                vscode.window.showInformationMessage('APG project compilation completed successfully');
                resolve();
            }
        });
    });
}

/**
 * Run the generated Flask-AppBuilder application
 */
async function runGeneratedApp(workspaceFolder: vscode.WorkspaceFolder): Promise<void> {
    const generatedDir = path.join(workspaceFolder.uri.fsPath, 'generated');
    const appFile = path.join(generatedDir, 'app.py');
    
    if (!fs.existsSync(appFile)) {
        vscode.window.showErrorMessage(
            'No generated application found. Compile the project first.',
            'Compile'
        ).then((selection) => {
            if (selection === 'Compile') {
                compileProject(workspaceFolder);
            }
        });
        return;
    }
    
    outputChannel.show(true);
    outputChannel.appendLine(`Starting APG application: ${appFile}`);
    
    // Create new terminal for running the app
    const terminal = vscode.window.createTerminal({
        name: 'APG Application',
        cwd: generatedDir
    });
    
    terminal.sendText('python app.py');
    terminal.show();
    
    // Show success message with URL
    vscode.window.showInformationMessage(
        'APG application started! Check the terminal for details.',
        'Open Browser'
    ).then((selection) => {
        if (selection === 'Open Browser') {
            vscode.env.openExternal(vscode.Uri.parse('http://localhost:8080'));
        }
    });
}

/**
 * Validate syntax of APG file
 */
async function validateSyntax(uri: vscode.Uri): Promise<void> {
    outputChannel.appendLine(`Validating APG syntax: ${uri.fsPath}`);
    
    return new Promise((resolve, reject) => {
        const workspaceDir = path.dirname(uri.fsPath);
        const command = `apg validate`;
        
        exec(command, { cwd: workspaceDir }, (error, stdout, stderr) => {
            if (error) {
                outputChannel.appendLine(`Validation failed: ${error.message}`);
                outputChannel.appendLine(stderr);
                vscode.window.showErrorMessage(`APG validation failed: ${error.message}`);
                reject(error);
            } else {
                outputChannel.appendLine('Validation successful!');
                outputChannel.appendLine(stdout);
                vscode.window.showInformationMessage('APG syntax validation passed');
                resolve();
            }
        });
    });
}

/**
 * Show preview of APG file
 */
async function showPreview(uri: vscode.Uri): Promise<void> {
    const panel = vscode.window.createWebviewPanel(
        'apgPreview',
        `APG Preview - ${path.basename(uri.fsPath)}`,
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.dirname(uri.fsPath))]
        }
    );
    
    // Read the APG file content
    const content = fs.readFileSync(uri.fsPath, 'utf8');
    
    // Generate preview HTML
    panel.webview.html = generatePreviewHTML(content, path.basename(uri.fsPath));
}

/**
 * Create new APG project
 */
async function createNewProject(): Promise<void> {
    const projectName = await vscode.window.showInputBox({
        prompt: 'Enter project name',
        placeHolder: 'my-apg-project',
        validateInput: (value) => {
            if (!value || value.trim().length === 0) {
                return 'Project name cannot be empty';
            }
            if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
                return 'Project name can only contain letters, numbers, hyphens, and underscores';
            }
            return null;
        }
    });
    
    if (!projectName) {
        return;
    }
    
    const workspaceFolder = getWorkspaceFolder();
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder is open');
        return;
    }
    
    const projectPath = path.join(workspaceFolder.uri.fsPath, projectName);
    
    return new Promise((resolve, reject) => {
        const command = `apg init ${projectName} --target flask-appbuilder`;
        
        exec(command, { cwd: workspaceFolder.uri.fsPath }, (error, stdout, stderr) => {
            if (error) {
                outputChannel.appendLine(`Project creation failed: ${error.message}`);
                outputChannel.appendLine(stderr);
                vscode.window.showErrorMessage(`Failed to create APG project: ${error.message}`);
                reject(error);
            } else {
                outputChannel.appendLine(`APG project created successfully: ${projectPath}`);
                outputChannel.appendLine(stdout);
                vscode.window.showInformationMessage(
                    `APG project '${projectName}' created successfully!`,
                    'Open Project'
                ).then((selection) => {
                    if (selection === 'Open Project') {
                        vscode.commands.executeCommand('vscode.openFolder', vscode.Uri.file(projectPath));
                    }
                });
                resolve();
            }
        });
    });
}

/**
 * Restart the language server
 */
async function restartLanguageServer(context: vscode.ExtensionContext): Promise<void> {
    outputChannel.appendLine('Restarting APG Language Server...');
    updateStatusBar('restarting');
    
    if (client) {
        await client.stop();
        client = undefined;
    }
    
    // Wait a moment before restarting
    setTimeout(() => {
        startLanguageServer(context);
    }, 1000);
}

/**
 * Show APG menu
 */
function showAPGMenu() {
    const items: vscode.QuickPickItem[] = [
        {
            label: '$(gear) Compile Current File',
            description: 'Compile the currently open APG file',
            detail: 'Ctrl+Shift+B'
        },
        {
            label: '$(folder) Compile Project',
            description: 'Compile entire APG project',
            detail: 'Build all APG files in the workspace'
        },
        {
            label: '$(play) Run Generated App',
            description: 'Run the generated Flask-AppBuilder application',
            detail: 'Ctrl+F5'
        },
        {
            label: '$(check) Validate Syntax',
            description: 'Validate APG syntax',
            detail: 'Ctrl+Shift+V'
        },
        {
            label: '$(preview) Show Preview',
            description: 'Show preview of APG file',
            detail: 'Visual representation of the APG code'
        },
        {
            label: '$(add) Create New Project',
            description: 'Create a new APG project',
            detail: 'Initialize a new APG project with templates'
        },
        {
            label: '$(refresh) Restart Language Server',
            description: 'Restart the APG Language Server',
            detail: 'Useful when language server stops responding'
        }
    ];
    
    vscode.window.showQuickPick(items, {
        placeHolder: 'Select an APG command'
    }).then((selection) => {
        if (!selection) return;
        
        switch (selection.label) {
            case '$(gear) Compile Current File':
                vscode.commands.executeCommand('apg.compile');
                break;
            case '$(folder) Compile Project':
                vscode.commands.executeCommand('apg.compileProject');
                break;
            case '$(play) Run Generated App':
                vscode.commands.executeCommand('apg.runGenerated');
                break;
            case '$(check) Validate Syntax':
                vscode.commands.executeCommand('apg.validateSyntax');
                break;
            case '$(preview) Show Preview':
                vscode.commands.executeCommand('apg.showPreview');
                break;
            case '$(add) Create New Project':
                vscode.commands.executeCommand('apg.createProject');
                break;
            case '$(refresh) Restart Language Server':
                vscode.commands.executeCommand('apg.restartLanguageServer');
                break;
        }
    });
}

/**
 * Update status bar item
 */
function updateStatusBar(status: string) {
    switch (status) {
        case 'connected':
            statusBarItem.text = "$(gear) APG";
            statusBarItem.tooltip = "APG Language Support - Connected";
            statusBarItem.backgroundColor = undefined;
            break;
        case 'compiling':
            statusBarItem.text = "$(sync~spin) APG";
            statusBarItem.tooltip = "APG Language Support - Compiling...";
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            break;
        case 'error':
            statusBarItem.text = "$(error) APG";
            statusBarItem.tooltip = "APG Language Support - Error";
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            break;
        case 'restarting':
            statusBarItem.text = "$(sync~spin) APG";
            statusBarItem.tooltip = "APG Language Support - Restarting...";
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            break;
        default:
            statusBarItem.text = "$(gear) APG";
            statusBarItem.tooltip = "APG Language Support";
            statusBarItem.backgroundColor = undefined;
    }
}

/**
 * Get current workspace folder
 */
function getWorkspaceFolder(): vscode.WorkspaceFolder | undefined {
    if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
        return vscode.workspace.workspaceFolders[0];
    }
    return undefined;
}

/**
 * Generate HTML for APG file preview
 */
function generatePreviewHTML(content: string, filename: string): string {
    const lines = content.split('\n');
    const highlightedContent = lines.map((line, index) => {
        const lineNumber = index + 1;
        const escapedLine = line
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        
        return `<div class="line"><span class="line-number">${lineNumber}</span><span class="line-content">${escapedLine}</span></div>`;
    }).join('');
    
    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>APG Preview - ${filename}</title>
        <style>
            body {
                font-family: 'Courier New', Consolas, monospace;
                margin: 0;
                padding: 20px;
                background-color: var(--vscode-editor-background);
                color: var(--vscode-editor-foreground);
            }
            .header {
                border-bottom: 1px solid var(--vscode-panel-border);
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .filename {
                font-size: 18px;
                font-weight: bold;
                color: var(--vscode-textLink-foreground);
            }
            .content {
                line-height: 1.5;
            }
            .line {
                display: flex;
                min-height: 20px;
            }
            .line-number {
                width: 50px;
                text-align: right;
                padding-right: 10px;
                color: var(--vscode-editorLineNumber-foreground);
                user-select: none;
                flex-shrink: 0;
            }
            .line-content {
                flex: 1;
                white-space: pre;
            }
            .line:hover {
                background-color: var(--vscode-editor-lineHighlightBackground);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="filename">${filename}</div>
        </div>
        <div class="content">
            ${highlightedContent}
        </div>
    </body>
    </html>
    `;
}