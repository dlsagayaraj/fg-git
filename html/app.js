var remote = require('electron').remote,
arguments = remote.getGlobal('sharedObject').prop1;
document.getElementById("user_name").innerHTML="Hello "+arguments[2]
