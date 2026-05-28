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
import { spawn } from 'child_process';
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

    const lintCommand = vscode.commands.registerCommand('apg.lint', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }

        await runAPGJsonCommand('Lint', editor.document.uri, ['lint', editor.document.uri.fsPath, '--json']);
    });

    const formatCommand = vscode.commands.registerCommand('apg.format', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }

        await runAPGTextCommand('Format', editor.document.uri, ['format', editor.document.uri.fsPath, '--write']);
    });

    const graphCommand = vscode.commands.registerCommand('apg.graph', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }

        await runAPGJsonCommand('Graph Suite', editor.document.uri, ['graph-suite', editor.document.uri.fsPath, '--json']);
    });

    const explainCommand = vscode.commands.registerCommand('apg.explain', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }

        const symbol = await vscode.window.showInputBox({
            prompt: 'APG symbol, diagnostic code, or handler target to explain',
            placeHolder: 'table.Customer, APG0100, or InvoiceForm.Save'
        });
        if (!symbol) {
            return;
        }

        const queryFlag = /^APG\d{4}$/i.test(symbol) ? '--diagnostic' : '--symbol';
        await runAPGJsonCommand('Explain', editor.document.uri, ['explain', editor.document.uri.fsPath, queryFlag, symbol, '--json']);
    });

    const packageCommand = vscode.commands.registerCommand('apg.package', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'apg') {
            vscode.window.showErrorMessage('No APG file is currently open');
            return;
        }

        const target = await vscode.window.showQuickPick(['web', 'desktop', 'mobile', 'container'], {
            placeHolder: 'Select APG package profile'
        });
        if (!target) {
            return;
        }

        const workspaceFolder = getWorkspaceFolder();
        const outDir = workspaceFolder
            ? path.join(workspaceFolder.uri.fsPath, 'dist', target)
            : path.join(path.dirname(editor.document.uri.fsPath), 'dist', target);
        await runAPGJsonCommand('Package', editor.document.uri, ['package', editor.document.uri.fsPath, '--target', target, '--out', outDir, '--json']);
    });

    const capabilitiesCommand = vscode.commands.registerCommand('apg.capabilities', async () => {
        await runAPGWorkspaceJsonCommand('Capability Contracts', ['capabilities', 'contracts', '--json']);
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
        lintCommand,
        formatCommand,
        graphCommand,
        explainCommand,
        packageCommand,
        capabilitiesCommand,
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

async function runAPGJsonCommand(label: string, uri: vscode.Uri, args: string[]): Promise<void> {
    const stdout = await runAPGCommand(args, path.dirname(uri.fsPath), label);
    await showJsonPreview(`${label} - ${path.basename(uri.fsPath)}`, stdout);
}

async function runAPGWorkspaceJsonCommand(label: string, args: string[]): Promise<void> {
    const workspaceFolder = getWorkspaceFolder();
    const cwd = workspaceFolder ? workspaceFolder.uri.fsPath : process.cwd();
    const stdout = await runAPGCommand(args, cwd, label);
    await showJsonPreview(label, stdout);
}

async function runAPGTextCommand(label: string, uri: vscode.Uri, args: string[]): Promise<void> {
    await runAPGCommand(args, path.dirname(uri.fsPath), label);
    vscode.window.showInformationMessage(`APG ${label.toLowerCase()} completed`);
}

function runAPGCommand(args: string[], cwd: string, label: string): Promise<string> {
    outputChannel.show(true);
    outputChannel.appendLine(`APG ${label}: apg ${args.join(' ')}`);

    return new Promise((resolve, reject) => {
        const child = spawn('apg', args, { cwd });
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            const text = data.toString();
            stdout += text;
            outputChannel.append(text);
        });
        child.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            outputChannel.append(text);
        });
        child.on('error', (error) => {
            updateStatusBar('error');
            vscode.window.showErrorMessage(`APG ${label} failed: ${error.message}`);
            reject(error);
        });
        child.on('close', (code) => {
            if (code === 0) {
                resolve(stdout);
            } else {
                updateStatusBar('error');
                const message = stderr.trim() || stdout.trim() || `exit code ${code}`;
                vscode.window.showErrorMessage(`APG ${label} failed: ${message}`);
                reject(new Error(message));
            }
        });
    });
}

async function showJsonPreview(title: string, jsonText: string): Promise<void> {
    const document = await vscode.workspace.openTextDocument({
        content: jsonText,
        language: 'json'
    });
    await vscode.window.showTextDocument(document, vscode.ViewColumn.Beside);
}

/**
 * Compile a single APG file
 */
async function compileFile(uri: vscode.Uri): Promise<void> {
    outputChannel.show(true);
    outputChannel.appendLine(`Compiling APG file: ${uri.fsPath}`);
    
    updateStatusBar('compiling');

    const workspaceDir = path.dirname(uri.fsPath);
    const outputDir = path.join(workspaceDir, 'generated');
    await runAPGCommand(['compile', uri.fsPath, '--target', 'python', '--output', outputDir, '--verify'], workspaceDir, 'Compilation');
    updateStatusBar('connected');
    vscode.window.showInformationMessage('APG compilation completed successfully');
}

/**
 * Compile entire APG project
 */
async function compileProject(workspaceFolder: vscode.WorkspaceFolder): Promise<void> {
    outputChannel.show(true);
    outputChannel.appendLine(`Compiling APG project: ${workspaceFolder.uri.fsPath}`);
    
    updateStatusBar('compiling');
    
    const apgFiles = await vscode.workspace.findFiles('**/*.apg', '**/{generated,dist,node_modules,.venv}/**', 1);
    if (apgFiles.length === 0) {
        vscode.window.showErrorMessage('No APG source file found in the workspace');
        updateStatusBar('error');
        return;
    }

    const sourceFile = apgFiles[0].fsPath;
    const outputDir = path.join(workspaceFolder.uri.fsPath, 'generated');
    await runAPGCommand(['compile', sourceFile, '--target', 'python', '--output', outputDir, '--verify'], workspaceFolder.uri.fsPath, 'Project compilation');
    updateStatusBar('connected');
    vscode.window.showInformationMessage('APG project compilation completed successfully');
}

/**
 * Run the generated Python application
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

    await runAPGJsonCommand('Validate', uri, ['validate', uri.fsPath, '--target', 'python', '--json']);
    vscode.window.showInformationMessage('APG validation passed');
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

    await runAPGCommand([
        'create',
        'project',
        '--name',
        projectName,
        '--template',
        'basic_agent',
        '--output',
        projectPath,
        '--no-interactive'
    ], workspaceFolder.uri.fsPath, 'Project creation');

    outputChannel.appendLine(`APG project created successfully: ${projectPath}`);
    vscode.window.showInformationMessage(
        `APG project '${projectName}' created successfully!`,
        'Open Project'
    ).then((selection) => {
        if (selection === 'Open Project') {
            vscode.commands.executeCommand('vscode.openFolder', vscode.Uri.file(projectPath));
        }
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
            description: 'Run the generated Python application',
            detail: 'Ctrl+F5'
        },
        {
            label: '$(check) Validate Syntax',
            description: 'Validate APG syntax',
            detail: 'Ctrl+Shift+V'
        },
        {
            label: '$(warning) Lint Current File',
            description: 'Run APG lint JSON contract',
            detail: 'apg lint --json'
        },
        {
            label: '$(symbol-keyword) Format Current File',
            description: 'Apply deterministic APG formatting',
            detail: 'apg format --write'
        },
        {
            label: '$(type-hierarchy) Show Graph Suite',
            description: 'Render APG graph evidence',
            detail: 'apg graph-suite --json'
        },
        {
            label: '$(info) Explain Symbol',
            description: 'Explain an APG symbol, diagnostic, or handler',
            detail: 'apg explain --json'
        },
        {
            label: '$(package) Package Current File',
            description: 'Build an APG package profile',
            detail: 'apg package --json'
        },
        {
            label: '$(library) Browse Capability Contracts',
            description: 'Inspect executable APG capability contracts',
            detail: 'apg capabilities contracts --json'
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
            case '$(warning) Lint Current File':
                vscode.commands.executeCommand('apg.lint');
                break;
            case '$(symbol-keyword) Format Current File':
                vscode.commands.executeCommand('apg.format');
                break;
            case '$(type-hierarchy) Show Graph Suite':
                vscode.commands.executeCommand('apg.graph');
                break;
            case '$(info) Explain Symbol':
                vscode.commands.executeCommand('apg.explain');
                break;
            case '$(package) Package Current File':
                vscode.commands.executeCommand('apg.package');
                break;
            case '$(library) Browse Capability Contracts':
                vscode.commands.executeCommand('apg.capabilities');
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
