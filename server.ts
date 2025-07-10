import { serve } from "bun";
import { join } from "path";

const publicDir = join(import.meta.dir, "."); // Serve from the root directory

serve({
  port: 4000,
  async fetch(request) {
    const url = new URL(request.url);
    let filePath = join(publicDir, url.pathname);

    // If the path is a directory, try to serve index.html or ppo.html
    if ((await Bun.file(filePath).exists()) && (await Bun.file(filePath).text()).length === 0) { // Check if it's a directory
      filePath = join(filePath, "index.html"); // Default to ppo.html for this specific request
    }

    const file = Bun.file(filePath);

    if (await file.exists()) {
      return new Response(file);
    } else {
      return new Response("Not Found", { status: 404 });
    }
  },
});

console.log("PPO GUI server running on http://localhost:4000");
