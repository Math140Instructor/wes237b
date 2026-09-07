#include <arpa/inet.h>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;

atomic<bool> running(true);

mutex streamMutex;

vector<int> browserClients;

vector<unsigned char> initSegment;

vector<unsigned char> currentFragment;

bool initReady = false;

// ============================================================
// Send entire buffer
// ============================================================

bool sendAll(
    int fd,
    const void *data,
    size_t size) {

  const char *ptr =
      static_cast<const char *>(data);

  size_t total = 0;

  while (total < size) {

    ssize_t sent =
        send(
            fd,
            ptr + total,
            size - total,
            MSG_NOSIGNAL);

    if (sent <= 0)
      return false;

    total +=
        static_cast<size_t>(sent);
  }

  return true;
}

// ============================================================
// Send one MP4 packet to browser
//
// HTTP body format:
//
// 4 byte packet length
// MP4 packet bytes
//
// HTTP itself is chunked.
// ============================================================

bool sendPacket(
    int fd,
    const vector<unsigned char> &data) {

  uint32_t size =
      static_cast<uint32_t>(
          data.size());

  unsigned char length[4] = {

      static_cast<unsigned char>(
          (size >> 24) & 0xff),

      static_cast<unsigned char>(
          (size >> 16) & 0xff),

      static_cast<unsigned char>(
          (size >> 8) & 0xff),

      static_cast<unsigned char>(
          size & 0xff)};

  size_t bodySize =
      4 + data.size();

  char chunkHeader[64];

  int headerLength =
      snprintf(
          chunkHeader,
          sizeof(chunkHeader),
          "%zx\r\n",
          bodySize);

  if (!sendAll(
          fd,
          chunkHeader,
          headerLength))
    return false;

  if (!sendAll(
          fd,
          length,
          sizeof(length)))
    return false;

  if (!data.empty() &&
      !sendAll(
          fd,
          data.data(),
          data.size()))
    return false;

  return sendAll(
      fd,
      "\r\n",
      2);
}

// ============================================================
// Read big-endian integers
// ============================================================

uint32_t readBE32(
    const unsigned char *p) {

  return

      (static_cast<uint32_t>(p[0]) << 24) |

      (static_cast<uint32_t>(p[1]) << 16) |

      (static_cast<uint32_t>(p[2]) << 8) |

      static_cast<uint32_t>(p[3]);
}

uint64_t readBE64(
    const unsigned char *p) {

  uint64_t value = 0;

  for (int i = 0; i < 8; i++) {

    value =
        (value << 8) |
        static_cast<uint64_t>(
            p[i]);
  }

  return value;
}

// ============================================================
// Remove browser
// ============================================================

void removeBrowserLocked(
    size_t index) {

  close(
      browserClients[index]);

  browserClients.erase(
      browserClients.begin() +
      static_cast<long>(index));
}

// ============================================================
// Broadcast latest MP4 fragment
// ============================================================

void broadcastLocked(
    const vector<unsigned char> &packet) {

  for (size_t i = 0;
       i < browserClients.size();) {

    if (!sendPacket(
            browserClients[i],
            packet)) {

      removeBrowserLocked(i);

    } else {

      i++;
    }
  }
}

// ============================================================
// Process MP4 box
// ============================================================

void processMp4Box(
    const vector<unsigned char> &box,
    const string &type) {

  lock_guard<mutex> lock(
      streamMutex);

  // ==========================================================
  // Initialization segment
  //
  // ftyp
  // moov
  // etc.
  //
  // Everything before first moof.
  // ==========================================================

  if (!initReady) {

    if (type == "moof") {

      initReady = true;

      cout << "MP4 init ready: "
           << initSegment.size()
           << " bytes\n";

      // Browser may have connected before camera.
      broadcastLocked(
          initSegment);

      currentFragment =
          box;

      return;
    }

    initSegment.insert(
        initSegment.end(),
        box.begin(),
        box.end());

    return;
  }

  // ==========================================================
  // New media fragment
  // ==========================================================

  if (type == "moof") {

    currentFragment.clear();

    currentFragment.insert(
        currentFragment.end(),
        box.begin(),
        box.end());

    return;
  }

  // ==========================================================
  // Finish fragment
  // ==========================================================

  if (!currentFragment.empty()) {

    currentFragment.insert(
        currentFragment.end(),
        box.begin(),
        box.end());

    if (type == "mdat") {

      broadcastLocked(
          currentFragment);

      currentFragment.clear();
    }
  }
}

// ============================================================
// Reset stream when camera reconnects
// ============================================================

void resetStream() {

  lock_guard<mutex> lock(
      streamMutex);

  initSegment.clear();

  currentFragment.clear();

  initReady = false;

  // Existing browsers must get a fresh MP4 initialization
  // segment after a camera restart.

  for (int fd :
       browserClients) {

    close(fd);
  }

  browserClients.clear();
}

// ============================================================
// Camera TCP server
//
// Receives fragmented MP4 directly from camera.cpp.
// ============================================================

void cameraServer() {

  int serverFd =
      socket(
          AF_INET,
          SOCK_STREAM,
          0);

  if (serverFd < 0) {

    perror("camera socket");

    exit(1);
  }

  int enable = 1;

  setsockopt(
      serverFd,
      SOL_SOCKET,
      SO_REUSEADDR,
      &enable,
      sizeof(enable));

  sockaddr_in address{};

  address.sin_family =
      AF_INET;

  address.sin_addr.s_addr =
      INADDR_ANY;

  address.sin_port =
      htons(5000);

  if (bind(
          serverFd,
          reinterpret_cast<
              sockaddr *>(
              &address),
          sizeof(address)) < 0) {

    perror("camera bind");

    close(serverFd);

    exit(1);
  }

  if (listen(
          serverFd,
          1) < 0) {

    perror("camera listen");

    close(serverFd);

    exit(1);
  }

  cout << "Waiting for camera on TCP :5000\n";

  while (running) {

    int cameraFd =
        accept(
            serverFd,
            nullptr,
            nullptr);

    if (cameraFd < 0)
      continue;

    // ========================================================
    // Disable Nagle buffering
    // ========================================================

    int one = 1;

    setsockopt(
        cameraFd,
        IPPROTO_TCP,
        TCP_NODELAY,
        &one,
        sizeof(one));

    // ========================================================
    // Don't allow huge amounts of stale video to sit inside
    // the receive buffer.
    // ========================================================

    int receiveBuffer =
        128 * 1024;

    setsockopt(
        cameraFd,
        SOL_SOCKET,
        SO_RCVBUF,
        &receiveBuffer,
        sizeof(receiveBuffer));

    cout << "Camera connected\n";

    resetStream();

    vector<unsigned char> pending;

    pending.reserve(
        2 * 1024 * 1024);

    unsigned char input[65536];

    while (running) {

      ssize_t received =
          recv(
              cameraFd,
              input,
              sizeof(input),
              0);

      if (received <= 0)
        break;

      pending.insert(
          pending.end(),
          input,
          input + received);

      // ======================================================
      // Extract complete MP4 boxes
      // ======================================================

      while (pending.size() >= 8) {

        uint64_t boxSize =
            readBE32(
                pending.data());

        size_t headerSize = 8;

        // Extended MP4 size
        if (boxSize == 1) {

          if (pending.size() < 16)
            break;

          boxSize =
              readBE64(
                  pending.data() + 8);

          headerSize = 16;
        }

        // Unknown streaming size.
        if (boxSize == 0)
          break;

        if (boxSize <
                headerSize ||

            boxSize >
                64ULL *
                    1024ULL *
                    1024ULL) {

          cerr
              << "Invalid MP4 box size: "
              << boxSize
              << "\n";

          close(cameraFd);

          cameraFd = -1;

          break;
        }

        if (pending.size() <
            boxSize) {

          break;
        }

        string type(

            reinterpret_cast<
                const char *>(
                pending.data() + 4),

            4);

        vector<unsigned char> box(

            pending.begin(),

            pending.begin() +
                static_cast<size_t>(
                    boxSize));

        pending.erase(

            pending.begin(),

            pending.begin() +
                static_cast<size_t>(
                    boxSize));

        processMp4Box(
            box,
            type);
      }

      if (cameraFd < 0)
        break;
    }

    if (cameraFd >= 0)
      close(cameraFd);

    cout << "Camera disconnected\n";
  }

  close(serverFd);
}

// ============================================================
// Browser HTML
// ============================================================

string htmlPage() {

  return R"HTML(
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1">

<title>RB3 Camera</title>

<style>

html,
body {

    margin: 0;

    width: 100%;
    height: 100%;

    background: black;

    overflow: hidden;
}

body {

    display: flex;

    align-items: center;

    justify-content: center;
}

video {

    width: 100%;
    height: 100%;

    object-fit: contain;

    background: black;
}

#status {

    position: fixed;

    top: 10px;
    left: 10px;

    padding: 6px 8px;

    background:
        rgba(0,0,0,.65);

    color: white;

    font-family:
        monospace;

    font-size: 13px;
}

</style>

</head>

<body>

<video
    id="video"
    autoplay
    muted
    playsinline>
</video>

<div id="status">
CONNECTING
</div>

<script>

const video =
    document.getElementById(
        "video");

const status =
    document.getElementById(
        "status");

// ============================================================
// Codec extraction
// ============================================================

function hex(value) {

    return value
        .toString(16)
        .padStart(2, "0")
        .toUpperCase();
}

function codecFromInit(init) {

    for (
        let i = 0;
        i + 8 < init.length;
        ++i
    ) {

        // avcC
        if (
            init[i]     === 0x61 &&
            init[i + 1] === 0x76 &&
            init[i + 2] === 0x63 &&
            init[i + 3] === 0x43
        ) {

            return (
                "avc1." +

                hex(
                    init[i + 5]) +

                hex(
                    init[i + 6]) +

                hex(
                    init[i + 7])
            );
        }
    }

    return "avc1.42E01F";
}

// ============================================================
// Read packet from HTTP stream
// ============================================================

async function nextPacket(
    reader,
    state) {

    // Need 4-byte size first.
    while (
        state.buffer.length < 4
    ) {

        const {
            value,
            done
        } =
            await reader.read();

        if (done)
            return null;

        const merged =
            new Uint8Array(

                state.buffer.length +

                value.length);

        merged.set(
            state.buffer);

        merged.set(
            value,
            state.buffer.length);

        state.buffer =
            merged;
    }

    const size =
        (
            state.buffer[0] *
                16777216 +

            state.buffer[1] *
                65536 +

            state.buffer[2] *
                256 +

            state.buffer[3]
        ) >>> 0;

    // Wait for whole packet.
    while (
        state.buffer.length <
        size + 4
    ) {

        const {
            value,
            done
        } =
            await reader.read();

        if (done)
            return null;

        const merged =
            new Uint8Array(

                state.buffer.length +

                value.length);

        merged.set(
            state.buffer);

        merged.set(
            value,
            state.buffer.length);

        state.buffer =
            merged;
    }

    const packet =
        state.buffer.slice(
            4,
            4 + size);

    state.buffer =
        state.buffer.slice(
            4 + size);

    return packet;
}

// ============================================================
// Start one stream connection
// ============================================================

async function runStream() {

    const response =
        await fetch(

            "/stream",

            {
                cache:
                    "no-store"
            });

    if (
        !response.ok ||
        !response.body
    ) {

        throw new Error(
            "Could not open stream");
    }

    const reader =
        response.body.getReader();

    const state = {

        buffer:
            new Uint8Array(0)
    };

    // ========================================================
    // MP4 initialization segment
    // ========================================================

    const init =
        await nextPacket(
            reader,
            state);

    if (!init) {

        throw new Error(
            "No MP4 initialization segment");
    }

    const codec =
        codecFromInit(init);

    const mime =
        `video/mp4; codecs="${codec}"`;

    console.log(
        "Video codec:",
        mime);

    if (
        !MediaSource.isTypeSupported(
            mime)
    ) {

        throw new Error(
            "Unsupported codec: " +
            mime);
    }

    // ========================================================
    // MediaSource
    // ========================================================

    const mediaSource =
        new MediaSource();

    video.src =
        URL.createObjectURL(
            mediaSource);

    await new Promise(
        resolve => {

            mediaSource
                .addEventListener(

                    "sourceopen",

                    resolve,

                    {
                        once: true
                    });
        });

    const sourceBuffer =
        mediaSource.addSourceBuffer(
            mime);

    sourceBuffer.mode =
        "segments";

    // ========================================================
    // Because camera.cpp uses key-int-max=1,
    // every fragment is independently decodable.
    //
    // Therefore stale queued fragments can be discarded.
    // ========================================================

    const queue = [
        init
    ];

    function pump() {

        if (
            sourceBuffer.updating ||
            queue.length === 0
        ) {

            return;
        }

        sourceBuffer.appendBuffer(
            queue.shift());
    }

    // ========================================================
    // Keep browser at live edge
    // ========================================================

    sourceBuffer
        .addEventListener(

            "updateend",

            () => {

                if (
                    video.buffered.length
                ) {

                    const end =
                        video.buffered.end(

                            video.buffered
                                .length - 1);

                    const start =
                        video.buffered.start(
                            0);

                    // ----------------------------------------
                    // If browser gets >80ms behind,
                    // jump nearly to live edge.
                    // ----------------------------------------

                    if (
                        end -
                        video.currentTime >
                        0.08
                    ) {

                        video.currentTime =
                            Math.max(

                                start,

                                end -
                                0.02);
                    }

                    // ----------------------------------------
                    // Keep only a very short history.
                    // ----------------------------------------

                    if (
                        end - start > 1.0 &&
                        !sourceBuffer.updating
                    ) {

                        try {

                            sourceBuffer.remove(

                                0,

                                Math.max(
                                    0,
                                    end - 0.50));

                        } catch (_) {
                        }
                    }
                }

                video
                    .play()
                    .catch(() => {});

                status.textContent =
                    "LIVE";

                pump();
            });

    sourceBuffer
        .addEventListener(

            "error",

            () => {

                status.textContent =
                    "BUFFER ERROR";
            });

    pump();

    // ========================================================
    // Receive live fragments
    // ========================================================

    while (true) {

        const fragment =
            await nextPacket(
                reader,
                state);

        if (!fragment) {

            throw new Error(
                "Camera disconnected");
        }

        // ====================================================
        // CRITICAL LOW-LATENCY BEHAVIOR
        //
        // Do not queue stale video.
        //
        // Since every frame is a keyframe, this is safe.
        // ====================================================

        if (queue.length > 0) {

            queue.length = 0;
        }

        queue.push(
            fragment);

        pump();
    }
}

// ============================================================
// Automatic reconnect
// ============================================================

async function start() {

    while (true) {

        try {

            status.textContent =
                "CONNECTING";

            await runStream();

        } catch (error) {

            console.error(
                error);

            status.textContent =
                "RECONNECTING";

            await new Promise(

                resolve =>

                    setTimeout(
                        resolve,
                        250));
        }
    }
}

start();

</script>

</body>

</html>
)HTML";
}

// ============================================================
// HTTP client
// ============================================================

void handleHttpClient(
    int fd) {

  char request[4096];

  ssize_t received =
      recv(
          fd,
          request,
          sizeof(request) - 1,
          0);

  if (received <= 0) {

    close(fd);

    return;
  }

  request[received] =
      '\0';

  string req(request);

  // ==========================================================
  // Video stream endpoint
  // ==========================================================

  if (
      req.find(
          "GET /stream ") !=
      string::npos
  ) {

    // --------------------------------------------------------
    // Disable Nagle buffering
    // --------------------------------------------------------

    int one = 1;

    setsockopt(
        fd,
        IPPROTO_TCP,
        TCP_NODELAY,
        &one,
        sizeof(one));

    // --------------------------------------------------------
    // Kill slow browser connections quickly instead of
    // allowing them to hold the camera streaming thread.
    // --------------------------------------------------------

    timeval timeout{};

    timeout.tv_sec =
        0;

    timeout.tv_usec =
        100000;

    setsockopt(
        fd,
        SOL_SOCKET,
        SO_SNDTIMEO,
        &timeout,
        sizeof(timeout));

    // --------------------------------------------------------
    // Keep browser-side TCP buffering small.
    // --------------------------------------------------------

    int sendBuffer =
        128 * 1024;

    setsockopt(
        fd,
        SOL_SOCKET,
        SO_SNDBUF,
        &sendBuffer,
        sizeof(sendBuffer));

    const string headers =

        "HTTP/1.1 200 OK\r\n"

        "Content-Type: "
        "application/octet-stream\r\n"

        "Cache-Control: "
        "no-store, no-cache, "
        "must-revalidate\r\n"

        "Pragma: no-cache\r\n"

        "Transfer-Encoding: "
        "chunked\r\n"

        "Connection: "
        "keep-alive\r\n"

        "\r\n";

    if (!sendAll(
            fd,
            headers.data(),
            headers.size())) {

      close(fd);

      return;
    }

    lock_guard<mutex> lock(
        streamMutex);

    // Browser connected after camera started.
    if (
        initReady &&
        !sendPacket(
            fd,
            initSegment)
    ) {

      close(fd);

      return;
    }

    browserClients.push_back(
        fd);

    cout
        << "Browser connected\n";

    return;
  }

  // ==========================================================
  // Web page
  // ==========================================================

  const string html =
      htmlPage();

  const string response =

      "HTTP/1.1 200 OK\r\n"

      "Content-Type: "
      "text/html; charset=utf-8\r\n"

      "Cache-Control: "
      "no-store\r\n"

      "Content-Length: " +
      to_string(
          html.size()) +

      "\r\n"

      "Connection: close\r\n"

      "\r\n" +

      html;

  sendAll(
      fd,
      response.data(),
      response.size());

  close(fd);
}

// ============================================================
// HTTP server
// ============================================================

void httpServer() {

  int serverFd =
      socket(
          AF_INET,
          SOCK_STREAM,
          0);

  if (serverFd < 0) {

    perror(
        "HTTP socket");

    exit(1);
  }

  int enable = 1;

  setsockopt(
      serverFd,
      SOL_SOCKET,
      SO_REUSEADDR,
      &enable,
      sizeof(enable));

  sockaddr_in address{};

  address.sin_family =
      AF_INET;

  address.sin_addr.s_addr =
      INADDR_ANY;

  address.sin_port =
      htons(8080);

  if (bind(
          serverFd,
          reinterpret_cast<
              sockaddr *>(
              &address),
          sizeof(address)) < 0) {

    perror(
        "HTTP bind");

    close(serverFd);

    exit(1);
  }

  if (listen(
          serverFd,
          16) < 0) {

    perror(
        "HTTP listen");

    close(serverFd);

    exit(1);
  }

  cout
      << "Browser server: "
      << "http://<RB3-IP>:8080/\n";

  while (running) {

    int fd =
        accept(
            serverFd,
            nullptr,
            nullptr);

    if (fd < 0)
      continue;

    thread(
        handleHttpClient,
        fd)
        .detach();
  }

  close(serverFd);
}

// ============================================================
// Main
// ============================================================

int main() {

  signal(
      SIGPIPE,
      SIG_IGN);

  cout
      << "=====================================\n";

  cout
      << "RB3 low-latency video server\n";

  cout
      << "Camera input : TCP 5000\n";

  cout
      << "Browser      : HTTP 8080\n";

  cout
      << "=====================================\n";

  thread cameraThread(
      cameraServer);

  thread webThread(
      httpServer);

  cameraThread.join();

  webThread.join();

  return 0;
}