import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { ReactElement } from 'react';
import ErrorBoundary from '../ErrorBoundary';

function ThrowingComponent(): ReactElement {
  throw new Error('Test error');
}

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div>Hello</div>
      </ErrorBoundary>
    );
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('renders error UI when child throws', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('An unexpected error occurred in this section.')).toBeInTheDocument();
    expect(screen.getByText('Test error')).toBeInTheDocument();
    spy.mockRestore();
  });

  it('shows "Try again" and "Reload page" buttons', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );
    expect(screen.getByText('Try again')).toBeInTheDocument();
    expect(screen.getByText('Reload page')).toBeInTheDocument();
    spy.mockRestore();
  });

  it('renders custom fallback when provided', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <ThrowingComponent />
      </ErrorBoundary>
    );
    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    spy.mockRestore();
  });

  it('logs error with scope label', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary scope="TestScope">
        <ThrowingComponent />
      </ErrorBoundary>
    );
    // componentDidCatch logs with scope
    const scopeCall = spy.mock.calls.find(
      (call) => typeof call[0] === 'string' && call[0].includes('[ErrorBoundary:TestScope]')
    );
    expect(scopeCall).toBeDefined();
    spy.mockRestore();
  });

  it('detects chunk loading errors and shows appropriate message', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    function ChunkErrorComponent(): ReactElement {
      throw new Error('Failed to fetch dynamically imported module /src/foo.js');
    }

    render(
      <ErrorBoundary>
        <ChunkErrorComponent />
      </ErrorBoundary>
    );
    expect(screen.getByText('Failed to load component')).toBeInTheDocument();
    expect(screen.getByText(/component failed to load/i)).toBeInTheDocument();
    spy.mockRestore();
  });

  it('"Try again" resets error state', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    let shouldThrow = true;

    function ConditionalThrow(): ReactElement {
      if (shouldThrow) throw new Error('Conditional error');
      return <div>Recovered</div>;
    }

    render(
      <ErrorBoundary>
        <ConditionalThrow />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    // Fix the underlying issue, then retry
    shouldThrow = false;
    fireEvent.click(screen.getByText('Try again'));

    expect(screen.getByText('Recovered')).toBeInTheDocument();
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    spy.mockRestore();
  });

  it('shows "Unable to recover" after 3 retries', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    function AlwaysThrows(): ReactElement {
      throw new Error('Persistent error');
    }

    render(
      <ErrorBoundary>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    // Retry 3 times — each retry re-throws
    for (let i = 0; i < 3; i++) {
      fireEvent.click(screen.getByText('Try again'));
    }

    expect(screen.getByText('Unable to recover')).toBeInTheDocument();
    expect(screen.getByText('Reload page')).toBeInTheDocument();
    expect(screen.queryByText('Try again')).not.toBeInTheDocument();
    spy.mockRestore();
  });
});
