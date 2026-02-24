const LOGOS = [
  {
    name: "SARC",
    src: "/logos/sarc.png",
    alt: "Satellite Application Research Center",
    href: "https://sarc.snu.ac.kr",
  },
  {
    name: "SATGeo",
    src: "/logos/satgeo.png",
    alt: "Satellite Geophysics Laboratory",
    href: "https://satgeo.snu.ac.kr",
  },
  {
    name: "SNU",
    src: "/logos/snu.png",
    alt: "Seoul National University",
    href: "https://snu.ac.kr",
  },
];

export default function Footer() {
  return (
    <footer className="border-t border-[#1e2a40] bg-[#080c14] px-4 py-2">
      <div className="flex items-center justify-center gap-8">
        {LOGOS.map((logo) => (
          <a
            key={logo.name}
            href={logo.href}
            target="_blank"
            rel="noopener noreferrer"
            title={logo.alt}
            className="opacity-60 hover:opacity-100 transition-opacity"
          >
            <img
              src={logo.src}
              alt={logo.alt}
              className="h-7 sm:h-8 w-auto"
            />
          </a>
        ))}
      </div>
    </footer>
  );
}
