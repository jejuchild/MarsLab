import { useEffect, useRef, useCallback, useState } from "react";

/* ================================================================
 * MARS DEFENDER — Canvas Space Shooter Easter Egg
 * ================================================================
 * Mars-themed space shooter. Defend your ship against Martian
 * asteroids and alien probes in the void above Mars.
 *
 * Arrow keys / WASD to move, Space to shoot, P to pause, Esc to quit.
 * ================================================================ */

interface SpaceGameProps {
  onClose: () => void;
}

/* ─── Constants ─── */
const SHIP_W = 36;
const SHIP_H = 28;
const BULLET_R = 2;
const BULLET_SPEED = 8;
const SHIP_SPEED = 5;
const STAR_COUNT = 120;
const ASTEROID_INTERVAL = 900;
const ALIEN_INTERVAL = 4000;
const POWERUP_INTERVAL = 12000;

/* ─── Mars palette ─── */
const MARS = {
  bg: "#0a0e17",
  ship: "#60a5fa",
  shipGlow: "#93c5fd",
  cockpit: "#bfdbfe",
  engine: "#fbbf24",
  // Asteroids — rusty Martian red/brown
  asteroidFill: "#b5451a",
  asteroidStroke: "#d46a3c",
  asteroidDark: "#7a2e10",
  // Alien probes — metallic silver/purple
  alienBody: "#a78bfa",
  alienDome: "#7c3aed",
  alienEye: "#22d3ee",
  alienStroke: "#c4b5fd",
  // Bullets
  playerBullet: "#fbbf24",
  alienBullet: "#ef4444",
  // Powerups
  rapid: "#f59e0b",
  shield: "#38bdf8",
  spread: "#f472b6",
  // UI
  primary: "#135bec",
  text: "#e2e8f0",
  textDim: "#64748b",
  danger: "#ef4444",
  marsOrb: "#c45427",
};

/* ─── Rank system ─── */
function getRank(score: number): string {
  if (score >= 5000) return "MARS DEFENDER";
  if (score >= 3000) return "COMMANDER";
  if (score >= 1500) return "CAPTAIN";
  if (score >= 800) return "LIEUTENANT";
  if (score >= 300) return "ENSIGN";
  return "CADET";
}

/* ─── Types ─── */
type GamePhase = "title" | "playing" | "gameover";
interface Star { x: number; y: number; r: number; speed: number; warm: boolean; }
interface Bullet { x: number; y: number; dy: number; isAlien?: boolean; }
interface Asteroid { x: number; y: number; r: number; speed: number; rot: number; drot: number; hp: number; seed: number; }
interface Alien { x: number; y: number; w: number; h: number; speed: number; dx: number; hp: number; shootCd: number; }
interface Particle { x: number; y: number; vx: number; vy: number; life: number; maxLife: number; color: string; r: number; }
interface PowerUp { x: number; y: number; speed: number; type: "rapid" | "shield" | "spread"; }
interface ScorePopup { x: number; y: number; text: string; life: number; color: string; }

export default function SpaceGame({ onClose }: SpaceGameProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fadeIn, setFadeIn] = useState(true);

  const stateRef = useRef({
    running: true,
    phase: "title" as GamePhase,
    score: 0,
    lives: 3,
    level: 1,
    ship: { x: 0, y: 0 },
    keys: new Set<string>(),
    bullets: [] as Bullet[],
    asteroids: [] as Asteroid[],
    aliens: [] as Alien[],
    particles: [] as Particle[],
    powerups: [] as PowerUp[],
    stars: [] as Star[],
    scorePopups: [] as ScorePopup[],
    lastAsteroid: 0,
    lastAlien: 0,
    lastPowerup: 0,
    lastShot: 0,
    shootCooldown: 150,
    rapidFireEnd: 0,
    spreadEnd: 0,
    shieldEnd: 0,
    shakeEnd: 0,
    damageFlashEnd: 0,
    levelUpEnd: 0,
    prevLevel: 1,
    kills: 0,
    titleBlink: 0,
  });

  // Fade in on mount
  useEffect(() => {
    const t = setTimeout(() => setFadeIn(false), 50);
    return () => clearTimeout(t);
  }, []);

  const spawnParticles = useCallback((x: number, y: number, count: number, color: string) => {
    const s = stateRef.current;
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1 + Math.random() * 3;
      s.particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 30 + Math.random() * 20,
        maxLife: 50,
        color,
        r: 1 + Math.random() * 2,
      });
    }
  }, []);

  const addScorePopup = useCallback((x: number, y: number, points: number) => {
    const s = stateRef.current;
    s.scorePopups.push({
      x, y,
      text: `+${points}`,
      life: 40,
      color: points >= 50 ? "#a78bfa" : "#fbbf24",
    });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width = window.innerWidth;
    const H = canvas.height = window.innerHeight;
    const s = stateRef.current;

    // Init
    s.ship = { x: W / 2, y: H - 80 };
    s.stars = Array.from({ length: STAR_COUNT }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: 0.3 + Math.random() * 1.5,
      speed: 0.2 + Math.random() * 0.8,
      warm: Math.random() > 0.7, // Some stars have warm Mars tint
    }));

    /* ─── Input ─── */
    let paused = false;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") { s.running = false; onClose(); return; }

      if (s.phase === "title") {
        if (e.key === " ") {
          e.preventDefault();
          s.phase = "playing";
          s.score = 0; s.lives = 3; s.level = 1; s.kills = 0; s.prevLevel = 1;
          s.bullets = []; s.asteroids = []; s.aliens = []; s.particles = []; s.powerups = []; s.scorePopups = [];
          s.ship = { x: W / 2, y: H - 80 };
          s.rapidFireEnd = 0; s.spreadEnd = 0; s.shieldEnd = 0;
          s.lastAsteroid = Date.now(); s.lastAlien = Date.now(); s.lastPowerup = Date.now();
        }
        return;
      }

      if ((e.key === "p" || e.key === "P") && s.phase === "playing") {
        paused = !paused;
        return;
      }
      if (e.key === "r" && s.phase === "gameover") {
        s.phase = "playing";
        paused = false;
        s.score = 0; s.lives = 3; s.level = 1; s.kills = 0; s.prevLevel = 1;
        s.bullets = []; s.asteroids = []; s.aliens = []; s.particles = []; s.powerups = []; s.scorePopups = [];
        s.ship = { x: W / 2, y: H - 80 };
        s.rapidFireEnd = 0; s.spreadEnd = 0; s.shieldEnd = 0;
        return;
      }
      s.keys.add(e.key);
    };
    const onKeyUp = (e: KeyboardEvent) => s.keys.delete(e.key);

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    /* ─── Drawing helpers ─── */
    function drawShip(cx: number, cy: number) {
      if (!ctx) return;
      const hasShield = Date.now() < s.shieldEnd;
      ctx.save();
      ctx.translate(cx, cy);
      // Main body
      ctx.beginPath();
      ctx.moveTo(0, -SHIP_H / 2);
      ctx.lineTo(-SHIP_W / 2, SHIP_H / 2);
      ctx.lineTo(-SHIP_W / 4, SHIP_H / 3);
      ctx.lineTo(SHIP_W / 4, SHIP_H / 3);
      ctx.lineTo(SHIP_W / 2, SHIP_H / 2);
      ctx.closePath();
      ctx.fillStyle = MARS.ship;
      ctx.fill();
      ctx.strokeStyle = MARS.shipGlow;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Cockpit
      ctx.beginPath();
      ctx.arc(0, -2, 4, 0, Math.PI * 2);
      ctx.fillStyle = MARS.cockpit;
      ctx.fill();
      // Engine glow
      ctx.beginPath();
      ctx.moveTo(-6, SHIP_H / 3);
      ctx.lineTo(0, SHIP_H / 2 + 6 + Math.random() * 4);
      ctx.lineTo(6, SHIP_H / 3);
      ctx.fillStyle = `rgba(251,191,36,${0.6 + Math.random() * 0.4})`;
      ctx.fill();
      ctx.restore();
      // Shield bubble
      if (hasShield) {
        ctx.beginPath();
        ctx.arc(cx, cy, SHIP_W * 0.7, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(56,189,248,${0.3 + Math.sin(Date.now() / 100) * 0.2})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    function drawAsteroid(a: Asteroid) {
      if (!ctx) return;
      ctx.save();
      ctx.translate(a.x, a.y);
      ctx.rotate(a.rot);
      ctx.beginPath();
      const sides = 7;
      for (let i = 0; i < sides; i++) {
        const ang = (i / sides) * Math.PI * 2;
        const wobble = a.r * (0.7 + 0.3 * Math.sin(i * 2.5 + a.seed));
        const px = Math.cos(ang) * wobble;
        const py = Math.sin(ang) * wobble;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      // Mars-red gradient
      const grad = ctx.createRadialGradient(-a.r * 0.2, -a.r * 0.2, 0, 0, 0, a.r);
      grad.addColorStop(0, MARS.asteroidStroke);
      grad.addColorStop(1, MARS.asteroidDark);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.strokeStyle = MARS.asteroidFill;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    }

    function drawAlien(a: Alien) {
      if (!ctx) return;
      ctx.save();
      ctx.translate(a.x, a.y);
      // Metallic probe body
      ctx.beginPath();
      ctx.ellipse(0, 0, a.w / 2, a.h / 3, 0, 0, Math.PI * 2);
      ctx.fillStyle = MARS.alienBody;
      ctx.fill();
      ctx.strokeStyle = MARS.alienStroke;
      ctx.lineWidth = 1;
      ctx.stroke();
      // Dome
      ctx.beginPath();
      ctx.ellipse(0, -a.h / 4, a.w / 4, a.h / 3, 0, Math.PI, 0);
      ctx.fillStyle = MARS.alienDome;
      ctx.fill();
      // Scanner eyes (cyan glow)
      ctx.shadowColor = MARS.alienEye;
      ctx.shadowBlur = 6;
      ctx.fillStyle = MARS.alienEye;
      ctx.fillRect(-5, -a.h / 4 - 2, 3, 3);
      ctx.fillRect(3, -a.h / 4 - 2, 3, 3);
      ctx.shadowBlur = 0;
      ctx.restore();
    }

    function drawPowerup(p: PowerUp) {
      if (!ctx) return;
      const colors = { rapid: MARS.rapid, shield: MARS.shield, spread: MARS.spread };
      const labels = { rapid: "R", shield: "S", spread: "W" };
      ctx.save();
      ctx.translate(p.x, p.y);
      // Glow
      ctx.shadowColor = colors[p.type];
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(0, 0, 10, 0, Math.PI * 2);
      ctx.fillStyle = colors[p.type];
      ctx.globalAlpha = 0.6 + Math.sin(Date.now() / 150) * 0.3;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(labels[p.type], 0, 0);
      ctx.restore();
    }

    function drawMiniShip(cx: number, cy: number) {
      if (!ctx) return;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(0.5, 0.5);
      ctx.beginPath();
      ctx.moveTo(0, -SHIP_H / 2);
      ctx.lineTo(-SHIP_W / 2, SHIP_H / 2);
      ctx.lineTo(SHIP_W / 2, SHIP_H / 2);
      ctx.closePath();
      ctx.fillStyle = MARS.ship;
      ctx.fill();
      ctx.restore();
    }

    function drawMarsOrb() {
      if (!ctx) return;
      // Subtle Mars planet in the background
      const orbX = W * 0.82;
      const orbY = H * 0.25;
      const orbR = Math.min(W, H) * 0.12;
      const grad = ctx.createRadialGradient(orbX - orbR * 0.3, orbY - orbR * 0.3, 0, orbX, orbY, orbR);
      grad.addColorStop(0, "rgba(196, 84, 39, 0.15)");
      grad.addColorStop(0.7, "rgba(196, 84, 39, 0.08)");
      grad.addColorStop(1, "rgba(196, 84, 39, 0)");
      ctx.beginPath();
      ctx.arc(orbX, orbY, orbR, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    function drawStars() {
      if (!ctx) return;
      for (const star of s.stars) {
        star.y += star.speed;
        if (star.y > H) { star.y = 0; star.x = Math.random() * W; }
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
        if (star.warm) {
          ctx.fillStyle = `rgba(255,200,150,${0.3 + star.r * 0.3})`;
        } else {
          ctx.fillStyle = `rgba(255,255,255,${0.3 + star.r * 0.3})`;
        }
        ctx.fill();
      }
    }

    /* ─── Title Screen ─── */
    function drawTitle() {
      if (!ctx) return;
      ctx.fillStyle = MARS.bg;
      ctx.fillRect(0, 0, W, H);
      drawStars();
      drawMarsOrb();

      s.titleBlink += 0.03;

      // Title text
      ctx.textAlign = "center";
      ctx.fillStyle = MARS.marsOrb;
      ctx.font = `bold ${Math.min(72, W * 0.06)}px monospace`;
      ctx.fillText("MARS DEFENDER", W / 2, H * 0.32);

      // Subtitle
      ctx.fillStyle = MARS.textDim;
      ctx.font = `${Math.min(16, W * 0.015)}px monospace`;
      ctx.fillText("Defend the orbital corridor above Mars", W / 2, H * 0.38);

      // Ship preview
      drawShip(W / 2, H * 0.52);

      // Blinking "PRESS SPACE" text
      const alpha = 0.4 + Math.sin(s.titleBlink) * 0.5;
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.fillStyle = MARS.text;
      ctx.font = `bold ${Math.min(20, W * 0.02)}px monospace`;
      ctx.fillText("PRESS SPACE TO LAUNCH", W / 2, H * 0.68);
      ctx.globalAlpha = 1;

      // Controls hint
      ctx.fillStyle = MARS.textDim;
      ctx.font = `${Math.min(13, W * 0.012)}px monospace`;
      ctx.fillText("Arrow keys / WASD to move  |  SPACE to shoot", W / 2, H * 0.78);
      ctx.fillText("P to pause  |  ESC to quit", W / 2, H * 0.82);

      // Version
      ctx.fillStyle = "rgba(100,116,139,0.4)";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText("MarsLab Easter Egg v1.0", W - 16, H - 12);
    }

    /* ─── Game Over Screen ─── */
    function drawGameOver() {
      if (!ctx) return;
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillRect(0, 0, W, H);

      ctx.textAlign = "center";

      // Title
      ctx.fillStyle = MARS.danger;
      ctx.font = `bold ${Math.min(48, W * 0.05)}px monospace`;
      ctx.fillText("MISSION FAILED", W / 2, H * 0.33);

      // Rank
      const rank = getRank(s.score);
      ctx.fillStyle = MARS.marsOrb;
      ctx.font = `bold ${Math.min(24, W * 0.025)}px monospace`;
      ctx.fillText(`Rank: ${rank}`, W / 2, H * 0.42);

      // Stats
      ctx.fillStyle = MARS.text;
      ctx.font = `${Math.min(18, W * 0.02)}px monospace`;
      ctx.fillText(`Score: ${s.score}  |  Kills: ${s.kills}  |  Level: ${s.level}`, W / 2, H * 0.50);

      // Controls
      ctx.fillStyle = MARS.textDim;
      ctx.font = `${Math.min(14, W * 0.015)}px monospace`;
      ctx.fillText("Press R to relaunch  |  ESC to quit", W / 2, H * 0.60);
    }

    /* ─── Game loop ─── */
    let animId = 0;

    function loop() {
      if (!s.running) return;
      animId = requestAnimationFrame(loop);
      if (!ctx) return;

      const now = Date.now();

      /* ── Title screen ── */
      if (s.phase === "title") {
        drawTitle();
        return;
      }

      /* ── Game Over screen ── */
      if (s.phase === "gameover") {
        drawGameOver();
        return;
      }

      /* ── Paused ── */
      if (paused) {
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = MARS.text;
        ctx.font = "bold 36px monospace";
        ctx.textAlign = "center";
        ctx.fillText("PAUSED", W / 2, H / 2);
        ctx.fillStyle = MARS.textDim;
        ctx.font = "16px monospace";
        ctx.fillText("Press P to resume", W / 2, H / 2 + 40);
        return;
      }

      /* ── Screen shake ── */
      const shaking = now < s.shakeEnd;
      if (shaking) {
        ctx.save();
        ctx.translate(
          (Math.random() - 0.5) * 6,
          (Math.random() - 0.5) * 6,
        );
      }

      /* ── Clear ── */
      ctx.fillStyle = MARS.bg;
      ctx.fillRect(0, 0, W, H);

      /* ── Background ── */
      drawStars();
      drawMarsOrb();

      /* ── Damage flash ── */
      if (now < s.damageFlashEnd) {
        const flashAlpha = (s.damageFlashEnd - now) / 200;
        ctx.fillStyle = `rgba(239,68,68,${flashAlpha * 0.15})`;
        ctx.fillRect(0, 0, W, H);
      }

      /* ── Level scaling ── */
      s.level = 1 + Math.floor(s.score / 500);
      if (s.level > s.prevLevel) {
        s.prevLevel = s.level;
        s.levelUpEnd = now + 2000;
      }
      const asteroidInterval = Math.max(300, ASTEROID_INTERVAL - s.level * 80);
      const alienInterval = Math.max(1500, ALIEN_INTERVAL - s.level * 300);

      /* ── Move ship ── */
      if (s.keys.has("ArrowLeft") || s.keys.has("a") || s.keys.has("A"))
        s.ship.x = Math.max(SHIP_W / 2, s.ship.x - SHIP_SPEED);
      if (s.keys.has("ArrowRight") || s.keys.has("d") || s.keys.has("D"))
        s.ship.x = Math.min(W - SHIP_W / 2, s.ship.x + SHIP_SPEED);
      if (s.keys.has("ArrowUp") || s.keys.has("w") || s.keys.has("W"))
        s.ship.y = Math.max(SHIP_H, s.ship.y - SHIP_SPEED);
      if (s.keys.has("ArrowDown") || s.keys.has("s") && !s.keys.has("Shift"))
        s.ship.y = Math.min(H - SHIP_H / 2, s.ship.y + SHIP_SPEED);

      /* ── Shoot ── */
      const cooldown = now < s.rapidFireEnd ? 70 : s.shootCooldown;
      if (s.keys.has(" ") && now - s.lastShot > cooldown) {
        s.lastShot = now;
        if (now < s.spreadEnd) {
          s.bullets.push({ x: s.ship.x, y: s.ship.y - SHIP_H / 2, dy: -BULLET_SPEED });
          s.bullets.push({ x: s.ship.x - 12, y: s.ship.y - SHIP_H / 2 + 4, dy: -BULLET_SPEED });
          s.bullets.push({ x: s.ship.x + 12, y: s.ship.y - SHIP_H / 2 + 4, dy: -BULLET_SPEED });
        } else {
          s.bullets.push({ x: s.ship.x, y: s.ship.y - SHIP_H / 2, dy: -BULLET_SPEED });
        }
      }

      /* ── Spawn asteroids ── */
      if (now - s.lastAsteroid > asteroidInterval) {
        s.lastAsteroid = now;
        const r = 12 + Math.random() * 22;
        s.asteroids.push({
          x: r + Math.random() * (W - r * 2),
          y: -r,
          r,
          speed: 1 + Math.random() * 2 + s.level * 0.3,
          rot: 0,
          drot: (Math.random() - 0.5) * 0.04,
          hp: r > 25 ? 2 : 1,
          seed: Math.random() * 10,
        });
      }

      /* ── Spawn aliens ── */
      if (now - s.lastAlien > alienInterval) {
        s.lastAlien = now;
        s.aliens.push({
          x: 40 + Math.random() * (W - 80),
          y: -30,
          w: 32 + Math.random() * 16,
          h: 20,
          speed: 1 + Math.random() * 1.5,
          dx: (Math.random() > 0.5 ? 1 : -1) * (1 + Math.random()),
          hp: 2 + Math.floor(s.level / 2),
          shootCd: 0,
        });
      }

      /* ── Spawn powerups ── */
      if (now - s.lastPowerup > POWERUP_INTERVAL) {
        s.lastPowerup = now;
        const types: Array<"rapid" | "shield" | "spread"> = ["rapid", "shield", "spread"];
        s.powerups.push({
          x: 30 + Math.random() * (W - 60),
          y: -12,
          speed: 1.5,
          type: types[Math.floor(Math.random() * types.length)],
        });
      }

      /* ── Update bullets ── */
      s.bullets = s.bullets.filter((b) => {
        b.y += b.dy;
        return b.y > -10 && b.y < H + 10;
      });

      /* ── Update asteroids ── */
      s.asteroids = s.asteroids.filter((a) => {
        a.y += a.speed;
        a.rot += a.drot;
        return a.y < H + a.r + 10;
      });

      /* ── Update aliens ── */
      for (const a of s.aliens) {
        a.y += a.speed;
        a.x += a.dx;
        if (a.x < a.w / 2 || a.x > W - a.w / 2) a.dx *= -1;
        a.shootCd -= 16;
        if (a.shootCd <= 0 && a.y > 20 && a.y < H * 0.6) {
          a.shootCd = 1200 + Math.random() * 800;
          s.bullets.push({ x: a.x, y: a.y + a.h / 2, dy: 4, isAlien: true });
        }
      }
      s.aliens = s.aliens.filter((a) => a.y < H + 40);

      /* ── Update powerups ── */
      s.powerups = s.powerups.filter((p) => {
        p.y += p.speed;
        return p.y < H + 20;
      });

      /* ── Update particles ── */
      s.particles = s.particles.filter((p) => {
        p.x += p.vx;
        p.y += p.vy;
        p.life--;
        return p.life > 0;
      });

      /* ── Update score popups ── */
      s.scorePopups = s.scorePopups.filter((p) => {
        p.y -= 1.2;
        p.life--;
        return p.life > 0;
      });

      /* ── Collision: player bullets vs asteroids ── */
      for (let bi = s.bullets.length - 1; bi >= 0; bi--) {
        const b = s.bullets[bi];
        if (b.isAlien) continue;
        for (let ai = s.asteroids.length - 1; ai >= 0; ai--) {
          const a = s.asteroids[ai];
          const dx = b.x - a.x, dy = b.y - a.y;
          if (dx * dx + dy * dy < a.r * a.r) {
            s.bullets.splice(bi, 1);
            a.hp--;
            if (a.hp <= 0) {
              spawnParticles(a.x, a.y, 12, MARS.asteroidStroke);
              s.asteroids.splice(ai, 1);
              s.score += 10;
              s.kills++;
              addScorePopup(a.x, a.y, 10);
            }
            break;
          }
        }
      }

      /* ── Collision: player bullets vs aliens ── */
      for (let bi = s.bullets.length - 1; bi >= 0; bi--) {
        const b = s.bullets[bi];
        if (b.isAlien) continue;
        for (let ai = s.aliens.length - 1; ai >= 0; ai--) {
          const a = s.aliens[ai];
          if (b.x > a.x - a.w / 2 && b.x < a.x + a.w / 2 && b.y > a.y - a.h / 2 && b.y < a.y + a.h / 2) {
            s.bullets.splice(bi, 1);
            a.hp--;
            if (a.hp <= 0) {
              spawnParticles(a.x, a.y, 18, MARS.alienBody);
              s.aliens.splice(ai, 1);
              s.score += 50;
              s.kills++;
              addScorePopup(a.x, a.y, 50);
            }
            break;
          }
        }
      }

      /* ── Collision: asteroids/aliens/alien-bullets vs ship ── */
      const hasShield = now < s.shieldEnd;
      const hitR = hasShield ? SHIP_W * 0.7 : SHIP_W * 0.4;
      // Asteroids
      for (let ai = s.asteroids.length - 1; ai >= 0; ai--) {
        const a = s.asteroids[ai];
        const dx = s.ship.x - a.x, dy = s.ship.y - a.y;
        if (dx * dx + dy * dy < (a.r + hitR) * (a.r + hitR)) {
          spawnParticles(a.x, a.y, 10, MARS.danger);
          s.asteroids.splice(ai, 1);
          if (!hasShield) {
            s.lives--;
            s.shakeEnd = now + 200;
            s.damageFlashEnd = now + 200;
            if (s.lives <= 0) s.phase = "gameover";
          }
        }
      }
      // Aliens
      for (let ai = s.aliens.length - 1; ai >= 0; ai--) {
        const a = s.aliens[ai];
        const dx = s.ship.x - a.x, dy = s.ship.y - a.y;
        if (Math.abs(dx) < (a.w / 2 + hitR) && Math.abs(dy) < (a.h / 2 + hitR)) {
          spawnParticles(a.x, a.y, 14, MARS.danger);
          s.aliens.splice(ai, 1);
          if (!hasShield) {
            s.lives--;
            s.shakeEnd = now + 200;
            s.damageFlashEnd = now + 200;
            if (s.lives <= 0) s.phase = "gameover";
          }
        }
      }
      // Alien bullets
      for (let bi = s.bullets.length - 1; bi >= 0; bi--) {
        const b = s.bullets[bi];
        if (!b.isAlien) continue;
        const dx = s.ship.x - b.x, dy = s.ship.y - b.y;
        if (dx * dx + dy * dy < hitR * hitR) {
          s.bullets.splice(bi, 1);
          if (!hasShield) {
            s.lives--;
            s.shakeEnd = now + 200;
            s.damageFlashEnd = now + 200;
            spawnParticles(s.ship.x, s.ship.y, 6, MARS.danger);
            if (s.lives <= 0) s.phase = "gameover";
          }
        }
      }

      /* ── Collision: ship vs powerups ── */
      for (let pi = s.powerups.length - 1; pi >= 0; pi--) {
        const p = s.powerups[pi];
        const dx = s.ship.x - p.x, dy = s.ship.y - p.y;
        if (dx * dx + dy * dy < (hitR + 10) * (hitR + 10)) {
          s.powerups.splice(pi, 1);
          if (p.type === "rapid") s.rapidFireEnd = now + 5000;
          if (p.type === "shield") s.shieldEnd = now + 6000;
          if (p.type === "spread") s.spreadEnd = now + 5000;
          spawnParticles(p.x, p.y, 8, MARS.engine);
        }
      }

      /* ─────── DRAW ─────── */

      // Asteroids
      for (const a of s.asteroids) drawAsteroid(a);

      // Aliens
      for (const a of s.aliens) drawAlien(a);

      // Powerups
      for (const p of s.powerups) drawPowerup(p);

      // Bullets
      for (const b of s.bullets) {
        ctx.beginPath();
        ctx.arc(b.x, b.y, BULLET_R, 0, Math.PI * 2);
        ctx.fillStyle = b.isAlien ? MARS.alienBullet : MARS.playerBullet;
        if (!b.isAlien) {
          ctx.shadowColor = MARS.playerBullet;
          ctx.shadowBlur = 4;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Particles
      for (const p of s.particles) {
        ctx.globalAlpha = p.life / p.maxLife;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Score popups
      for (const p of s.scorePopups) {
        ctx.globalAlpha = p.life / 40;
        ctx.fillStyle = p.color;
        ctx.font = "bold 14px monospace";
        ctx.textAlign = "center";
        ctx.fillText(p.text, p.x, p.y);
      }
      ctx.globalAlpha = 1;

      // Ship
      drawShip(s.ship.x, s.ship.y);

      // Restore after shake
      if (shaking) ctx.restore();

      /* ── HUD ── */
      ctx.fillStyle = MARS.text;
      ctx.font = "bold 16px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`SCORE  ${s.score}`, 16, 30);
      ctx.fillText(`LEVEL  ${s.level}`, 16, 52);
      ctx.fillText(`KILLS  ${s.kills}`, 16, 74);

      // Lives
      ctx.textAlign = "right";
      for (let i = 0; i < s.lives; i++) {
        drawMiniShip(W - 20 - i * 28, 28);
      }

      // Active power-ups
      ctx.textAlign = "left";
      let puY = 96;
      if (now < s.rapidFireEnd) {
        ctx.fillStyle = MARS.rapid; ctx.font = "12px monospace";
        ctx.fillText(`RAPID FIRE ${Math.ceil((s.rapidFireEnd - now) / 1000)}s`, 16, puY); puY += 18;
      }
      if (now < s.shieldEnd) {
        ctx.fillStyle = MARS.shield; ctx.font = "12px monospace";
        ctx.fillText(`SHIELD ${Math.ceil((s.shieldEnd - now) / 1000)}s`, 16, puY); puY += 18;
      }
      if (now < s.spreadEnd) {
        ctx.fillStyle = MARS.spread; ctx.font = "12px monospace";
        ctx.fillText(`SPREAD ${Math.ceil((s.spreadEnd - now) / 1000)}s`, 16, puY);
      }

      // Level-up notification
      if (now < s.levelUpEnd) {
        const alpha = Math.min(1, (s.levelUpEnd - now) / 1000);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = MARS.marsOrb;
        ctx.font = "bold 28px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`LEVEL ${s.level}`, W / 2, H * 0.15);
        ctx.globalAlpha = 1;
      }

      // Controls hint (first few seconds)
      if (s.score === 0 && now - s.lastAsteroid < 5000) {
        ctx.fillStyle = "rgba(148,163,184,0.5)";
        ctx.font = "13px monospace";
        ctx.textAlign = "center";
        ctx.fillText("Arrow keys / WASD to move  |  SPACE to shoot  |  P pause  |  ESC quit", W / 2, H - 20);
      }
    }

    animId = requestAnimationFrame(loop);

    return () => {
      s.running = false;
      cancelAnimationFrame(animId);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [onClose, spawnParticles, addScorePopup]);

  return (
    <div
      className="fixed inset-0 z-[200] bg-[#0a0e17] transition-opacity duration-300"
      style={{ opacity: fadeIn ? 0 : 1 }}
    >
      <canvas
        ref={canvasRef}
        className="block w-full h-full"
        tabIndex={0}
        autoFocus
      />
    </div>
  );
}
