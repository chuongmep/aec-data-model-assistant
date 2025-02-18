class AECDataModelClient {
    constructor(accessToken) {
        this.accessToken = accessToken;
    }

    async #submit(query, variables) {
        const resp = await fetch("https://developer.api.autodesk.com/aec/graphql", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${this.accessToken}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query, variables })
        });
        if (!resp.ok) {
            throw new Error(await resp.text());
        }
        const { data, errors } = await resp.json();
        if (errors) {
            throw new Error(errors.map(err => err.message).join("\n"));
        }
        return data;
    }

    async getHubs() {
        // TODO: add pagination
        const query = `query GetHubs {
            hubs {
                results {
                    id
                    name
                }
            }
        }`;
        const data = await this.#submit(query);
        return data.hubs.results.map(hub => ({ ...hub, type: "hub" }));
    }

    async getProjects(hubId) {
        // TODO: add pagination
        const query = `query GetProjects($hubId: ID!) {
            projects(hubId: $hubId) {
                results {
                    id
                    name
                }
            }
        }`;
        const data = await this.#submit(query, { hubId });
        return data.projects.results.map(project => ({ ...project, type: "project" }));
    }

    async getElementGroups(projectId) {
        // TODO: add pagination
        const query = `query GetElementGroups($projectId: ID!) {
            elementGroupsByProject(projectId: $projectId) {
                results {
                    id
                    name
                    alternativeIdentifiers {
                        fileVersionUrn
                    }
                }
            }
        }`;
        const data = await this.#submit(query, { projectId });
        return data.elementGroupsByProject.results.map(elementGroup => ({
            id: elementGroup.id,
            name: elementGroup.name,
            versionId: elementGroup.alternativeIdentifiers.fileVersionUrn,
            type: "elementgroup"
        }));
    }
}

export async function initBrowser(credentials, onSelectionChanged) {
    const client = new AECDataModelClient(credentials.access_token);
    const hubs = await client.getHubs();
    const $tree = document.querySelector("#browser > sl-tree");
    for (const hub of hubs) {
        $tree.append(createTreeItem(`hub|${hub.id}`, hub.name, "cloud", true));
    }
    $tree.addEventListener("sl-selection-change", function ({ detail }) {
        if (detail.selection.length === 1 && detail.selection[0].id.startsWith("itm|")) {
            debugger;
            const [, hubId, projectId, itemId, versionId] = detail.selection[0].id.split("|");
            const urn = btoa(versionId).replaceAll("=", "").replaceAll("/", "_");
            onSelectionChanged({ hubId, projectId, itemId, versionId, urn });
        }
    });

    function createTreeItem(id, text, icon, children = false) {
        const item = document.createElement("sl-tree-item");
        item.id = id;
        item.innerHTML = `<sl-icon name="${icon}"></sl-icon><span style="white-space: nowrap">${text}</span>`;
        if (children) {
            item.lazy = true;
            item.addEventListener("sl-lazy-load", async function (ev) {
                ev.stopPropagation();
                item.lazy = false;
                const tokens = item.id.split("|");
                switch (tokens[0]) {
                    case "hub": {
                        const projects = await client.getProjects(tokens[1]);
                        item.append(...projects.map(project => createTreeItem(`prj|${tokens[1]}|${project.id}`, project.name, "building", true)));
                        break;
                    }
                    case "prj": {
                        const elementGroups = await client.getElementGroups(tokens[2]);
                        item.append(...elementGroups.map(elementGroup => createTreeItem(`itm|${tokens[1]}|${tokens[2]}|${elementGroup.id}|${elementGroup.versionId}`, elementGroup.name, "boxes", false)));
                        break;
                    }
                }
            });
        }
        return item;
    }
}