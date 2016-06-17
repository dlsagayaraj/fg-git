const {app, BrowserWindow} = require('electron');
global.sharedObject = {prop1: process.argv}
var remote = require('electron').remote,
arguments = remote.getGlobal('sharedObject').prop1;
alert(arguments);



let mainWindow;

// Quit when all windows are closed.
app.on('window-all-closed', function() {
  app.quit();
});

// This method will be called when Electron has done everything
// initialization and ready for creating browser windows.
app.on('ready', function() {
  // Create the browser window.
  mainWindow = new BrowserWindow({width: 600, height: 600, frame: false});

  // and load the index.html of the app.
  mainWindow.loadURL('file://' + __dirname + '/index.html');
  setTimeout(function(){ app.quit(); }, 30000);

  // Emitted when the window is closed.
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });
});
