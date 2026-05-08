function copyBibTeX() {
  const bibtexElement = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const copyText = button ? button.querySelector(".copy-text") : null;

  if (!bibtexElement || !button || !copyText) {
    return;
  }

  const text = bibtexElement.textContent;

  navigator.clipboard.writeText(text).then(
    function () {
      button.classList.add("copied");
      copyText.textContent = "Copied";

      setTimeout(function () {
        button.classList.remove("copied");
        copyText.textContent = "Copy";
      }, 2000);
    },
    function () {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);

      button.classList.add("copied");
      copyText.textContent = "Copied";
      setTimeout(function () {
        button.classList.remove("copied");
        copyText.textContent = "Copy";
      }, 2000);
    }
  );
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

function updateScrollButton() {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (!scrollButton) {
    return;
  }

  if (window.pageYOffset > 300) {
    scrollButton.classList.add("visible");
  } else {
    scrollButton.classList.remove("visible");
  }
}

function updateSidebarProgress() {
  const progressFill = document.getElementById("sidebar-progress-fill");
  if (!progressFill) {
    return;
  }

  const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
  const scrollTop = window.scrollY || window.pageYOffset;
  const progress = documentHeight > 0 ? Math.min(scrollTop / documentHeight, 1) : 0;
  progressFill.style.height = `${progress * 100}%`;
}

const FINDING_SUMMARY_BOXES = [
  {
    left: 0.615,
    top: 0.013972055888223553,
    width: 0.37125,
    height: 0.1217564870259481,
    tone: "green"
  },
  {
    left: 0.615,
    top: 0.17564870259481039,
    width: 0.37125,
    height: 0.11776447105788423,
    tone: "red"
  },
  {
    left: 0.615,
    top: 0.32335329341317365,
    width: 0.37125,
    height: 0.14171656686626746,
    tone: "red"
  },
  {
    left: 0.615,
    top: 0.7984031936127745,
    width: 0.37125,
    height: 0.1536926147704591,
    tone: "red"
  }
];

function setupFindingHighlights() {
  const boxes = Array.from(document.querySelectorAll("[data-finding-highlight]"));

  boxes.forEach(function (box) {
    const index = Number(box.getAttribute("data-finding-highlight"));
    const highlight = FINDING_SUMMARY_BOXES[index];

    if (!highlight) {
      return;
    }

    box.style.left = `${highlight.left * 100}%`;
    box.style.top = `${highlight.top * 100}%`;
    box.style.width = `${highlight.width * 100}%`;
    box.style.height = `${highlight.height * 100}%`;
    box.dataset.highlightTone = highlight.tone;
  });
}

function updateFindingHighlights(activeIndex) {
  document.querySelectorAll("[data-finding-highlight]").forEach(function (box) {
    const index = Number(box.getAttribute("data-finding-highlight"));
    box.classList.toggle("is-active", index === activeIndex);
  });
}

function updateCarouselWindowHeight(carousel, slides, currentIndex) {
  const windowElement = carousel.querySelector(".slider-window");
  const activeSlide = slides[currentIndex];

  if (!windowElement || !activeSlide) {
    return;
  }

  windowElement.style.height = `${activeSlide.offsetHeight}px`;
}

function setupSectionObservers() {
  const sections = Array.from(document.querySelectorAll(".tracked-section"));
  const navLinks = Array.from(document.querySelectorAll("[data-section-link]"));
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!sections.length) {
    return;
  }

  const setActiveSection = function (sectionId) {
    navLinks.forEach(function (link) {
      const isActive = link.getAttribute("data-section-link") === sectionId;
      link.classList.toggle("is-active", isActive);
    });
  };

  if (prefersReducedMotion) {
    sections.forEach(function (section) {
      section.classList.add("is-visible");
    });
  } else {
    const revealObserver = new IntersectionObserver(
      function (entries, observer) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.18,
        rootMargin: "0px 0px -10% 0px"
      }
    );

    sections.forEach(function (section) {
      revealObserver.observe(section);
    });
  }

  const activeObserver = new IntersectionObserver(
    function (entries) {
      const visibleEntries = entries
        .filter(function (entry) {
          return entry.isIntersecting;
        })
        .sort(function (a, b) {
          return b.intersectionRatio - a.intersectionRatio;
        });

      if (visibleEntries.length > 0) {
        setActiveSection(visibleEntries[0].target.id);
      }
    },
    {
      threshold: [0.2, 0.35, 0.5, 0.7],
      rootMargin: "-10% 0px -45% 0px"
    }
  );

  sections.forEach(function (section) {
    activeObserver.observe(section);
  });

  if (sections[0]) {
    setActiveSection(sections[0].id);
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatTuple(label, values) {
  return `<span class="case-meta-pill"><strong>${label}=</strong>${escapeHtml(values)}</span>`;
}

function formatAgreementTrajectory(judgeScores) {
  if (!Array.isArray(judgeScores) || !judgeScores.length) {
    return "";
  }

  return `
    <div class="agreement-trajectory" aria-label="Agreement trajectory">
      ${judgeScores
        .map(function (score, index) {
          return `
            <span class="agreement-chip agreement-chip-${score}">
              <span class="agreement-chip-round">R${index + 1}</span>
              <span class="agreement-chip-score">${escapeHtml(score)}</span>
            </span>
          `;
        })
        .join("")}
    </div>
  `;
}

function formatUtterances(utterances) {
  if (!Array.isArray(utterances) || !utterances.length) {
    return '<p class="transcript-empty">No public turns available.</p>';
  }

  const visibleTurns = utterances.slice(0, 4);
  const rounds = [];

  for (let index = 0; index < visibleTurns.length; index += 2) {
    const roundNumber = Math.floor(index / 2) + 1;
    rounds.push(`
      <div class="utterance-round">
        <div class="utterance-round-label">Round ${roundNumber}</div>
        <div class="utterance-row">
          <span class="utterance-speaker">A1</span>
          <p class="utterance-text">${escapeHtml(visibleTurns[index] || "")}</p>
        </div>
        <div class="utterance-row">
          <span class="utterance-speaker">A2</span>
          <p class="utterance-text">${escapeHtml(visibleTurns[index + 1] || "")}</p>
        </div>
      </div>
    `);
  }

  return rounds.join("");
}

function renderCaseCard(caseData) {
  const preferenceTuple = `(${caseData.agent_1_topic_response}, ${caseData.agent_2_topic_response})`;
  const opennessTuple = `(${caseData.agent_1_persuadability}, ${caseData.agent_2_persuadability})`;
  const agreementTuple = `(${caseData.judge_scores.join(", ")})`;

  return `
    <article class="qual-case-card">
      <div class="qual-case-header">
        <span class="qual-case-slot">${escapeHtml(caseData.slot_label)}</span>
        <h4 class="qual-case-topic">${escapeHtml(caseData.topic_title)}</h4>
        <p class="qual-case-model">${escapeHtml(caseData.model_name)}</p>
      </div>

      <div class="qual-case-metadata">
        ${formatTuple("P", preferenceTuple)}
        ${formatTuple("O", opennessTuple)}
        ${formatTuple("A", agreementTuple)}
      </div>

      <div class="qual-case-agreement">
        <span class="agreement-label">Agreement trajectory</span>
        ${formatAgreementTrajectory(caseData.judge_scores)}
      </div>

      <dl class="qual-case-summaries">
        <div>
          <dt>Agent 1</dt>
          <dd>${escapeHtml(caseData.agent_1_summary)}</dd>
        </div>
        <div>
          <dt>Agent 2</dt>
          <dd>${escapeHtml(caseData.agent_2_summary)}</dd>
        </div>
      </dl>

      <div class="qual-transcript">
        <h5>Conversation</h5>
        <div class="utterance-list">
          ${formatUtterances(caseData.utterances)}
        </div>
      </div>
    </article>
  `;
}

function renderComparisonSlide(group) {
  const cards = Array.isArray(group.cases)
    ? group.cases
        .filter(function (caseData) {
          return caseData.display_ready === true &&
            caseData.judge_reasonable === true &&
            Number(caseData.near_duplicate_turns || 0) === 0;
        })
        .slice(0, 2)
        .map(function (caseData) {
          return renderCaseCard(caseData);
        })
    : [];

  return `
    <article class="qual-slide">
      <div class="qual-slide-header">
        <span class="qual-finding-id">${escapeHtml(group.finding_title)}</span>
        <h3 class="qual-slide-title">${escapeHtml(group.comparison_title)}</h3>
        <p class="qual-slide-claim"><strong>Claim:</strong> ${escapeHtml(group.claim)}</p>
        <p class="qual-slide-relationship"><strong>Relationship:</strong> ${escapeHtml(group.relationship)}</p>
      </div>

      <div class="qual-comparison-grid">
        ${cards.join("")}
      </div>
    </article>
  `;
}

function initializeCarousel(carousel) {
  if (!carousel || carousel.dataset.carouselInitialized === "true") {
    return;
  }

  const track = carousel.querySelector("[data-carousel-track]");
  const slides = track ? Array.from(track.children) : [];
  const dotsContainer = carousel.querySelector("[data-carousel-dots]");
  const prevButton = carousel.querySelector("[data-carousel-prev]");
  const nextButton = carousel.querySelector("[data-carousel-next]");
  const label = carousel.getAttribute("data-carousel-label") || "slide";
  const isFindingCarousel = label === "finding";

  if (!track || !slides.length || !dotsContainer || !prevButton || !nextButton) {
    return;
  }

  let currentIndex = 0;

  const goToSlide = function (index) {
    currentIndex = (index + slides.length) % slides.length;
    carousel.dataset.currentIndex = String(currentIndex);
    track.style.transform = `translateX(-${currentIndex * 100}%)`;
    updateCarouselWindowHeight(carousel, slides, currentIndex);

    Array.from(dotsContainer.children).forEach(function (dot, dotIndex) {
      dot.classList.toggle("is-active", dotIndex === currentIndex);
      dot.setAttribute("aria-current", dotIndex === currentIndex ? "true" : "false");
    });

    if (isFindingCarousel) {
      updateFindingHighlights(currentIndex);
    }
  };

  slides.forEach(function (_, index) {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "slider-dot";
    dot.setAttribute("aria-label", `Go to ${label} ${index + 1}`);
    dot.addEventListener("click", function () {
      goToSlide(index);
    });
    dotsContainer.appendChild(dot);
  });

  prevButton.addEventListener("click", function () {
    goToSlide(currentIndex - 1);
  });

  nextButton.addEventListener("click", function () {
    goToSlide(currentIndex + 1);
  });

  slides.forEach(function (slide) {
    slide.querySelectorAll("img").forEach(function (image) {
      image.addEventListener("load", function () {
        updateCarouselWindowHeight(carousel, slides, currentIndex);
      });
    });
  });

  carousel.dataset.carouselInitialized = "true";
  goToSlide(0);
}

function setupCarousels() {
  document.querySelectorAll("[data-carousel]").forEach(function (carousel) {
    initializeCarousel(carousel);
  });
}

async function renderQualitativeExamples() {
  const root = document.getElementById("qualitative-examples-root");

  if (!root) {
    return;
  }

  try {
    const comparisonsResponse = await fetch("selected_instances/website_comparisons.json");

    if (!comparisonsResponse.ok) {
      throw new Error("Unable to load qualitative comparison data.");
    }

    const comparisons = await comparisonsResponse.json();

    root.innerHTML = `
      <div class="qualitative-carousel" data-carousel data-carousel-label="qualitative example">
        <div class="qualitative-carousel-header">
          <div class="slider-controls slider-controls-top">
            <button class="slider-arrow" type="button" data-carousel-prev aria-label="Previous qualitative example">
              <i class="fas fa-arrow-left"></i>
            </button>
            <div class="slider-dots" data-carousel-dots></div>
            <button class="slider-arrow" type="button" data-carousel-next aria-label="Next qualitative example">
              <i class="fas fa-arrow-right"></i>
            </button>
          </div>
        </div>
        <div class="slider-window qualitative-window">
          <div class="slider-track qualitative-track" data-carousel-track>
            ${comparisons.map(function (group) {
              return renderComparisonSlide(group);
            }).join("")}
          </div>
        </div>
      </div>
    `;

    setupCarousels();
  } catch (error) {
    root.innerHTML = `
      <div class="qualitative-error">
        Unable to load qualitative examples. Serve the site through a local web server so the JSON files under <code>selected_instances/</code> can be fetched.
      </div>
    `;
  }
}

document.addEventListener("DOMContentLoaded", function () {
  setupFindingHighlights();
  setupSectionObservers();
  setupCarousels();
  renderQualitativeExamples();
  updateScrollButton();
  updateSidebarProgress();
});

window.addEventListener("scroll", function () {
  updateScrollButton();
  updateSidebarProgress();
});

window.addEventListener("resize", function () {
  document.querySelectorAll("[data-carousel][data-carousel-initialized='true']").forEach(function (carousel) {
    const track = carousel.querySelector("[data-carousel-track]");
    const slides = track ? Array.from(track.children) : [];
    const currentIndex = Number(carousel.dataset.currentIndex || 0);

    if (slides.length) {
      updateCarouselWindowHeight(carousel, slides, currentIndex);
    }
  });
  updateSidebarProgress();
});
