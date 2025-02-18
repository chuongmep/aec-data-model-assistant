import { authenticate, login, logout } from "./auth.js";
import { initBrowser } from "./browser.js";
import { initViewer, loadModel } from "./viewer.js";
import { initChatbot } from "./chatbot.js";

const credentials = await authenticate();
const $login = document.querySelector("#login");
$login.style.visibility = "visible";
if (credentials) {
    $login.innerText = "Logout";
    $login.onclick = () => logout();
    const viewer = await initViewer(credentials);
    await initBrowser(credentials, (el) => {
        loadModel(viewer, el.urn);
        initChatbot(credentials, el.itemId);
        document.getElementById("chatbot").addEventListener("click", function ({ target }) {
            if (target.dataset.guid) {
                viewer.model.getExternalIdMapping((mapping) => {
                    const dbid = mapping[target.dataset.guid];
                    viewer.isolate([dbid]);
                    viewer.fitToView([dbid]);
                });
            }
        });
    });
} else {
    $login.innerText = "Login";
    $login.onclick = () => login();
}